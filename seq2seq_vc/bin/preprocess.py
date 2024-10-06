#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Perform preprocessing and raw feature extraction."""

import argparse
import logging
import os

import librosa
import numpy as np
import soundfile as sf

import yaml

from tqdm import tqdm

from seq2seq_vc.datasets import AudioSCPDataset
from seq2seq_vc.utils import write_hdf5

import torch
from s3prl.nn import Featurizer
import s3prl_vc.models
from s3prl_vc.upstream.interface import get_upstream


def logmelfilterbank(
    audio,
    sampling_rate,
    fft_size=1024,
    hop_size=256,
    win_length=None,
    window="hann",
    num_mels=80,
    fmin=None,
    fmax=None,
    eps=1e-10,
    log_base=10.0,
):
    """Compute log-Mel filterbank feature.

    Args:
        audio (ndarray): Audio signal (T,).
        sampling_rate (int): Sampling rate.
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length. If set to None, it will be the same as fft_size.
        window (str): Window function type.
        num_mels (int): Number of mel basis.
        fmin (int): Minimum frequency in mel basis calculation.
        fmax (int): Maximum frequency in mel basis calculation.
        eps (float): Epsilon value to avoid inf in log calculation.
        log_base (float): Log base. If set to None, use np.log.

    Returns:
        ndarray: Log Mel filterbank feature (#frames, num_mels).

    """
    # get amplitude spectrogram
    x_stft = librosa.stft(
        audio,
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=win_length,
        window=window,
        pad_mode="reflect",
    )
    spc = np.abs(x_stft).T  # (#frames, #bins)

    # get mel basis
    fmin = 0 if fmin is None else fmin
    fmax = sampling_rate / 2 if fmax is None else fmax
    mel_basis = librosa.filters.mel(
        sr=sampling_rate,
        n_fft=fft_size,
        n_mels=num_mels,
        fmin=fmin,
        fmax=fmax,
    )
    mel = np.maximum(eps, np.dot(spc, mel_basis.T))

    if log_base is None:
        return np.log(mel)
    elif log_base == 10.0:
        return np.log10(mel)
    elif log_base == 2.0:
        return np.log2(mel)
    else:
        raise ValueError(f"{log_base} is not supported.")


def main():
    """Run preprocessing process."""
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess audio and then extract features (See detail in"
            " bin/preprocess.py)."
        )
    )
    parser.add_argument(
        "--wav-scp",
        "--scp",
        required=True,
        type=str,
        help="kaldi-style wav.scp file.",
    )
    parser.add_argument(
        "--segments",
        default=None,
        type=str,
        help=(
            "kaldi-style segments file. if use, you must to specify both scp and"
            " segments."
        ),
    )
    parser.add_argument(
        "--dumpdir",
        type=str,
        required=True,
        help="directory to dump feature files.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="yaml format configuration file.",
    )
    parser.add_argument(
        "--feat_layer",
        required=False,
        help="layer to extract features from.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    args = parser.parse_args()

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # load config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))

    dataset = AudioSCPDataset(
        args.wav_scp,
        segments=args.segments,
        return_utt_id=True,
        return_sampling_rate=True,
    )

    # check directly existence
    if not os.path.exists(args.dumpdir):
        os.makedirs(args.dumpdir, exist_ok=True)

    # load upstream extractor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractors = {}
    for feat_type in config.get("feat_list", ["mel", "hubert", "ppg_sxliu"]):
        if feat_type == "mel":
            extractor = {}
        elif feat_type == "encodec":
            from seq2seq_vc.utils.encodec import get_encodec_model, encodec_encode
            from encodec.utils import convert_audio

            extractor = {"model": get_encodec_model()}
        else:
            # if has checkpoint in config, load from checkpoint
            if "checkpoint" in config["feat_list"][feat_type].keys():
                checkpoint = config["feat_list"][feat_type]["checkpoint"]
                upstream_model = get_upstream(feat_type).to(device)
                upstream_model.eval()
                upstream_featurizer = Featurizer(upstream_model).to(device)
                upstream_featurizer.load_state_dict(
                    torch.load(checkpoint, map_location="cpu")["featurizer"]
                )
                logging.info(f"Loaded {feat_type} extractor parameters from {checkpoint}.")
            else:
                upstream_model = get_upstream(feat_type).to(device)
                upstream_featurizer = Featurizer(upstream_model).to(device)
                logging.info(f"Loaded {feat_type} extractor parameters from {feat_type} pretrained model.")
            upstream_featurizer.eval()

            extractor = {"model": upstream_model, "featurizer": upstream_featurizer}

        extractors[feat_type] = extractor
    # process each data
    for utt_id, (audio, fs) in tqdm(dataset):
        # check
        assert len(audio.shape) == 1, f"{utt_id} seems to be multi-channel signal."
        assert (
            np.abs(audio).max() <= 1.0
        ), f"{utt_id} seems to be different from 16 bit PCM."

        # resample to specified sampling rate in config
        if fs != config["sampling_rate"]:
            audio = librosa.resample(
                audio,
                orig_sr=fs,
                target_sr=config["sampling_rate"],
            )

        # trim silence
        if config["trim_silence"]:
            audio, _ = librosa.effects.trim(
                audio,
                top_db=config["trim_threshold_in_db"],
                frame_length=config["trim_frame_size"],
                hop_length=config["trim_hop_size"],
            )

        if "sampling_rate_for_feats" not in config:
            x = audio
            sampling_rate = config["sampling_rate"]
            hop_size = config["hop_size"]
        else:
            # NOTE(kan-bayashi): this procedure enables to train the model with different
            #   sampling rate for feature and audio, e.g., training with mel extracted
            #   using 16 kHz audio and 24 kHz audio as a target waveform
            x = librosa.resample(
                audio,
                orig_sr=fs,
                target_sr=config["sampling_rate_for_feats"],
            )
            sampling_rate = config["sampling_rate_for_feats"]
            assert config["hop_size"] * config["sampling_rate_for_feats"] % fs == 0, (
                "hop_size must be int value. please check sampling_rate_for_feats is"
                " correct."
            )
            hop_size = config["hop_size"] * config["sampling_rate_for_feats"] // fs

        # make sure the audio length and feature length are matched
        audio = np.pad(audio, (0, config["fft_size"]), mode="reflect")
        # audio = audio[: len(mel) * config["hop_size"]]
        # assert len(mel) * config["hop_size"] == len(audio)

        # apply global gain
        if config["global_gain_scale"] > 0.0:
            audio *= config["global_gain_scale"]
        if np.abs(audio).max() > 1.0:
            logging.warn(
                f"{utt_id} causes clipping (max: {np.abs(audio).max()}). "
                "it is better to re-consider global gain scale."
            )
            continue

        # save waveform
        if config["format"] == "hdf5":
            write_hdf5(
                os.path.join(args.dumpdir, f"{utt_id}.h5"),
                "wave",
                audio.astype(np.float32),
            )
        else:
            raise ValueError("support only hdf5 format.")

        # extract and save feature
        for feat_type in extractors:
            if feat_type == "mel":
                feat = logmelfilterbank(
                    x,
                    sampling_rate=sampling_rate,
                    hop_size=hop_size,
                    fft_size=config["fft_size"],
                    win_length=config["win_length"],
                    window=config["window"],
                    num_mels=config["num_mels"],
                    fmin=config["fmin"],
                    fmax=config["fmax"],
                    # keep compatibility
                    log_base=config.get("log_base", 10.0),
                )  # [n_frames, n_dim]
            elif feat_type == "encodec":
                encodec_model = extractors[feat_type]["model"]
                audio_for_encodec = convert_audio(
                    torch.from_numpy(x).unsqueeze(0),
                    sampling_rate,
                    encodec_model.sample_rate,
                    encodec_model.channels,
                )
                feat = encodec_encode(
                    audio_for_encodec, encodec_model
                )  # a list of [1, 128, T]
                feat = torch.concat(feat, dim=2).squeeze(0).numpy().T  # [T, 128]
            elif feat_type == "ppg_sxliu":
                with torch.no_grad():
                    xs = torch.from_numpy(x).unsqueeze(0).float().to(device)
                    ilens = torch.LongTensor([x.shape[0]]).to(device)

                    all_hs, all_hlens = extractors[feat_type]["model"](xs, ilens)
                    hs, _ = extractors[feat_type]["featurizer"](all_hs, all_hlens)
                    feat = hs[0].cpu().numpy()
            else:
                # for hubert only now
                with torch.no_grad():
                    xs = torch.from_numpy(x).unsqueeze(0).float().to(device)
                    ilens = torch.LongTensor([x.shape[0]]).to(device)

                    all_hs, all_hlens = extractors[feat_type]["model"](xs, ilens)
                    # for hubert_feature
                    if feat_type == "hubert":
                        if "feat_layer" in config and config["feat_layer"] is not None:
                            config["feat_layer"] = int(config["feat_layer"])
                            hs = all_hs[config["feat_layer"]]
                        else:
                            # get average HuBERT
                            hs, _ = extractors[feat_type]["featurizer"](all_hs, all_hlens)
                    feat = hs[0].cpu().numpy()

            write_hdf5(
                os.path.join(args.dumpdir, f"{utt_id}.h5"),
                feat_type,
                feat.astype(np.float32),
            )


if __name__ == "__main__":
    main()
