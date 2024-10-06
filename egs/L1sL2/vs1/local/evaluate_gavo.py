#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Modified by Haopeng Geng, 2024, The University of Tokyo
# Copyright 2022 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

import argparse
import multiprocessing as mp
import os

import numpy as np
import librosa

import torch
import torchaudio
from tqdm import tqdm
import yaml
import soundfile as sf

from seq2seq_vc.utils import find_files
from seq2seq_vc.utils.types import str2bool
from seq2seq_vc.evaluate.dtw_based import calculate_mcd_f0
from seq2seq_vc.evaluate.asr import load_asr_model, transcribe, calculate_measures
import evaluate
bertscore_model = evaluate.load("bertscore")


def get_basename(path):
    return os.path.splitext(os.path.split(path)[-1])[0]

def _calculate_asr_score(model, device, file_list, groundtruths):
    keys = ["hits", "substitutions",  "deletions", "insertions"]
    ers = {}
    c_results = {k: 0 for k in keys}
    w_results = {k: 0 for k in keys}

    for i, cvt_wav_path in enumerate(tqdm(file_list)):
        basename = get_basename(cvt_wav_path)
        groundtruth = groundtruths[basename] # get rid of the first character "E"
        
        # load waveform
        wav, _ = librosa.load(cvt_wav_path, sr=16000)

        # trascribe
        transcription = transcribe(model, device, wav)

        # error calculation
        c_result, w_result, norm_groundtruth, norm_transcription = calculate_measures(groundtruth, transcription)

        # if bert score is available
        ers[basename] = [c_result["cer"] * 100.0, w_result["wer"] * 100.0, norm_transcription, norm_groundtruth]

        for k in keys:
            c_results[k] += c_result[k]
            w_results[k] += w_result[k]
  
    # calculate over whole set
    def er(r):
        return float(r["substitutions"] + r["deletions"] + r["insertions"]) \
            / float(r["substitutions"] + r["deletions"] + r["hits"]) * 100.0

    cer = er(c_results)
    wer = er(w_results)

    return ers, cer, wer

def _calculate_asr_score_bert_score(model, device, file_list, groundtruths, bertscore_model):
    keys = ["hits", "substitutions",  "deletions", "insertions"]
    ers = {}
    bertscore_results = []
    c_results = {k: 0 for k in keys}
    w_results = {k: 0 for k in keys}

    for i, cvt_wav_path in enumerate(file_list):
        basename = get_basename(cvt_wav_path)
        groundtruth = groundtruths[basename] # get rid of the first character "E"
        
        # load waveform
        wav, _ = librosa.load(cvt_wav_path, sr=16000)

        # trascribe
        transcription = transcribe(model, device, wav)

        # error calculation
        c_result, w_result, norm_groundtruth, norm_transcription = calculate_measures(groundtruth, transcription)

        # if bert score is available
        bertscores = bertscore_model.compute(predictions=[norm_groundtruth], references=[norm_transcription], lang="en")
        
        ers[basename] = [c_result["cer"] * 100.0, w_result["wer"] * 100.0, norm_transcription, norm_groundtruth]
        ers[basename].append(bertscores["f1"][0])

        bertscore_results.append(bertscores["f1"][0])
        
        for k in keys:
            c_results[k] += c_result[k]
            w_results[k] += w_result[k]
  
    # calculate over whole set
    def er(r):
        return float(r["substitutions"] + r["deletions"] + r["insertions"]) \
            / float(r["substitutions"] + r["deletions"] + r["hits"]) * 100.0

    cer = er(c_results)
    wer = er(w_results)
    bertscore_mean = np.mean(bertscore_results)
    return ers, cer, wer, bertscore_mean

def _calculate_mcd_f0(file_list, gt_root, segments, trgspk, f0min, f0max, results, gv=False):
    for i, cvt_wav_path in tqdm(enumerate(file_list)):
        basename = get_basename(cvt_wav_path)
        
        # get ground truth target wav path
        gt_wav_path = os.path.join(gt_root, basename + ".wav")

        # read both converted and ground truth wav
        cvt_wav, cvt_fs = librosa.load(cvt_wav_path, sr=None)
        if segments is not None:
            gt_wav, gt_fs = librosa.load(gt_wav_path, sr=None,
                                         offset=segments[basename]["offset"],
                                         duration=segments[basename]["duration"]
                                         )
        else:
            gt_wav, gt_fs = librosa.load(gt_wav_path, sr=None)
        if cvt_fs != gt_fs:
            cvt_wav = torchaudio.transforms.Resample(cvt_fs, gt_fs)(torch.from_numpy(cvt_wav)).numpy()

        # calculate MCD, F0RMSE, F0CORR and DDUR
        res = calculate_mcd_f0(cvt_wav, gt_wav, gt_fs, f0min, f0max, calculate_gv=gv)

        results.append([basename, res])


def calculate_speechbertscore(ref_wav, gen_wav, model):
    ref_wav, _ = sf.read(ref_wav)
    gen_wav, _ = sf.read(gen_wav)
    
    precision, recall, f1_score = model.score(ref_wav, gen_wav)
    # Precision is the best metric base on paper:
    return precision

def _calculate_speechbertscore(file_list, gt_root, results, speechbertscore_model):
    for i, cvt_wav_path in tqdm(enumerate(file_list)):
        basename = get_basename(cvt_wav_path)
        
        # get ground truth target wav path
        gt_wav_path = os.path.join(gt_root, basename + ".wav")

        # calculate SPEECHBERTSCORE
        res = calculate_speechbertscore(gt_wav_path, cvt_wav_path, speechbertscore_model)
        
        results.append([basename, res])

def get_parser():
    parser = argparse.ArgumentParser(description="objective evaluation script.")
    parser.add_argument("--wavdir", required=True, type=str, help="directory for converted waveforms")
    parser.add_argument("--trgspk", required=True, type=str, help="target speaker")
    parser.add_argument("--data_root", type=str, default="./data", help="directory of data")
    parser.add_argument("--transcription", type=str, default="text", help="transcription file")
    parser.add_argument("--segments", type=str, default=None, help="segments file")
    parser.add_argument("--f0_path", required=True, type=str, help="yaml file storing f0 ranges")
    parser.add_argument("--n_jobs", default=10, type=int, help="number of parallel jobs")
    parser.add_argument("--gv", default=False, type=str2bool, help="calculate GV or not")
    parser.add_argument("--asr", default=True, type=str2bool, help="calculate ASR score or not")
    parser.add_argument("--mcd", default=False, type=str2bool, help="calculate MCD score or not")
    parser.add_argument("--bertscore", default=False, type=str2bool, help="calculate BERT score or not")
    parser.add_argument("--speechbertscore", default=False, type=str2bool, help="calculate SpeechBERT score or not")
    return parser


def main():
    args = get_parser().parse_args()

    trgspk = args.trgspk
    try:
        gt_root = os.path.join(args.data_root, "wav")
        # gt_root exists
        assert os.path.exists(gt_root)
    except:
        gt_root = args.data_root
        assert os.path.exists(gt_root)
    try:
        transcription_path = os.path.join(args.data_root, "etc", "arctic.data") # for arctic data
        assert os.path.exists(transcription_path)
    except:
        transcription_path = os.path.join(args.data_root, args.transcription) # for normal data
        assert os.path.exists(transcription_path)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # load f0min and f0 max
    with open(args.f0_path, 'r') as f:
        f0_all = yaml.load(f, Loader=yaml.FullLoader)
    f0min = f0_all[trgspk]["f0min"]
    f0max = f0_all[trgspk]["f0max"]

    # load ground truth transcriptions
    with open(transcription_path, "r") as f:
        lines = f.read().splitlines()
    groundtruths = {line.split(" ")[1]: " ".join(line.split(" ")[2:-1]).replace('"', '') for line in lines}

    # load segments if provided
    if args.segments is not None:
        with open(args.segments, "r") as f:
            lines = f.read().splitlines()
        segments = {}
        for line in lines:
            _id, _, start, end = line.split(" ")
            segments[_id] = {
                "offset": float(start),
                "duration": float(end) - float(start)
            }
    else:
        segments = None

    # find converted files
    converted_files = sorted(find_files(args.wavdir, query="*.wav"))
    print("number of utterances = {}".format(len(converted_files)))

    ##############################
    results = []
    # load ASR model
    if args.asr:
        asr_model = load_asr_model(device)
        # calculate error rates
        if args.bertscore:
            print("Calculating ASR-based score with BERTScore...")
            import evaluate
            bertscore_model = evaluate.load("bertscore")
            ers, cer, wer, bertscoremean = _calculate_asr_score_bert_score(asr_model, device, converted_files, groundtruths, bertscore_model=bertscore_model)
        else:
            print("Calculating ASR-based score...")
            ers, cer, wer = _calculate_asr_score(asr_model, device, converted_files, groundtruths)
    
    if args.speechbertscore:
        print("Calculating SpeechBERTScore...")
        SpeechBERTScore_model_type = "wavlm-large"
        from discrete_speech_metrics import SpeechBERTScore
        SpeechBERTScore_model = SpeechBERTScore(
            sr=16000,
            model_type=SpeechBERTScore_model_type,
            use_gpu=True)
        SpeechBERTScore_results = []
        _calculate_speechbertscore(converted_files, gt_root, SpeechBERTScore_results, SpeechBERTScore_model)
    # import pdb; pdb.set_trace()
    ##############################
    if args.mcd:
        print("Calculating MCD and f0-related scores...")
        # Get and divide list
        # file_lists = np.array_split(converted_files, args.n_jobs)
        # file_lists = [f_list.tolist() for f_list in file_lists]

        # multi processing
        # with mp.Manager() as manager:
        #     results = manager.list()
        #     processes = []
        #     for f in file_lists:
        #         p = mp.Process(
        #             target=_calculate_mcd_f0,
        #             args=(f, gt_root, segments, trgspk, f0min, f0max, results, args.gv),
        #         )
        #         p.start()
        #         processes.append(p)
        _calculate_mcd_f0(converted_files, gt_root, segments, trgspk, f0min, f0max, results, args.gv)
            # wait for all process
            # for p in processes:
            #     p.join()
    
    # results = [[x] for x in ers.keys()]
    
    # sorted_results = sorted(results, key=lambda x:x[0])
    # for result in sorted_results:
    #     d["basename"] = result[0]
    #     if args.mcd:
    #         d = {k: v for k, v in result[1].items()}
    #     if args.asr:
    #         d["CER"] = ers[result[0]][0]
    #         d["WER"] = ers[result[0]][1]
    #         d["GT_TRANSCRIPTION"] = ers[result[0]][2]
    #         d["CV_TRANSCRIPTION"] = ers[result[0]][3]
    #     if args.bertscore:
    #         d["CER"] = ers[result[0]][0]
    #         d["WER"] = ers[result[0]][1]
    #         d["GT_TRANSCRIPTION"] = ers[result[0]][2]
    #         d["CV_TRANSCRIPTION"] = ers[result[0]][3]
    #         d["BERTSCORE"] = ers[result[0]][4]
    #     if args.speechbertscore:
            
    #         d["SPEECHBERTSCORE"] = SpeechBERTScore_results[result[0]]
    #     results.append(d)
    if len(results) != 0:
        sorted_results = sorted(results, key=lambda x:x[0])
    else:
        sorted_results = [[x] for x in ers.keys()]
    
    for result in sorted_results:
        try:
            d = {k: v for k, v in result[1].items()}
        except:
            d = {}
        if args.asr:
            d["basename"] = result[0]
            d["CER"] = ers[result[0]][0]
            d["WER"] = ers[result[0]][1]
            d["GT_TRANSCRIPTION"] = ers[result[0]][2]
            d["CV_TRANSCRIPTION"] = ers[result[0]][3]
        if args.bertscore:
            d["basename"] = result[0]
            d["CER"] = ers[result[0]][0]
            d["WER"] = ers[result[0]][1]
            d["GT_TRANSCRIPTION"] = ers[result[0]][2]
            d["CV_TRANSCRIPTION"] = ers[result[0]][3]
            d["BERTSCORE"] = ers[result[0]][4]
        results.append(d)

    # utterance wise result
    for result in results:
        # get valid keys
        keys = list(result.keys())
        
        line = ""
        
        for k in keys:
            if type(result[k]) == float:
                line += f"{result[k]:.4f}, "
            elif type(result[k]) == str:
                line += f"{ result[k]} |"
        
        print(line)
                
                   
    
    # average result
    # mMCD = np.mean(np.array([result["MCD"] for result in results]))
    # mf0RMSE = np.mean(np.array([result["F0RMSE"] for result in results]))
    # mf0CORR = np.mean(np.array([result["F0CORR"] for result in results]))
    # mDDUR = np.mean(np.array([result["DDUR"] for result in results]))
    mCER = cer 
    mWER = wer
    mBERTSCORE = np.mean(SpeechBERTScore_results) if args.speechbertscore else None
    
    if args.bertscore:
        mBERTSCORE = np.mean(np.array([result["BERTSCORE"] for result in results]))
        print(
            "Mean CER, WER, BERTSCORE: {:.3f} {:.3f} {:.3f}".format(
                mCER, mWER, mBERTSCORE
            )
        )
        pass
    elif not args.gv:
        print(
            "Mean MCD, f0RMSE, f0CORR, DDUR, CER, WER: {:.2f} {:.2f} {:.3f} {:.3f} {:.1f} {:.1f}".format(
                mMCD, mf0RMSE, mf0CORR, mDDUR, mCER, mWER
            )
        )
    else:
        mGV = np.mean(np.array([result["GV"] for result in results]))
        print(
            "Mean MCD, GV, f0RMSE, f0CORR, DDUR, CER, WER: {:.2f} {:.3f} {:.2f} {:.3f} {:.3f} {:.1f} {:.1f}".format(
                mMCD, mGV, mf0RMSE, mf0CORR, mDDUR, mCER, mWER
            )
        )


if __name__ == "__main__":
    main()