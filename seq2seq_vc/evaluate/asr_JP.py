import editdistance as ed
import jiwer
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, Wav2Vec2Processor
from datasets import load_dataset   
import librosa
import pdb

ASR_PRETRAINED_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-japanese"

def load_asr_model(device):
    """Load model"""
    print(f"[INFO]: Load the pre-trained ASR by {ASR_PRETRAINED_MODEL}.")
    model = Wav2Vec2ForCTC.from_pretrained(ASR_PRETRAINED_MODEL).to(device)
    tokenizer = Wav2Vec2Tokenizer.from_pretrained(ASR_PRETRAINED_MODEL)
    processor = Wav2Vec2Processor.from_pretrained(ASR_PRETRAINED_MODEL)
    models = {"model": model, "tokenizer": tokenizer, "processor": processor}
    return models


def normalize_sentence(sentence):
    """Normalize sentence"""
    # Convert all characters to upper.
    sentence = sentence.upper()
    # Delete punctuations.
    sentence = jiwer.RemovePunctuation()(sentence)
    # Remove \n, \t, \r, \x0c.
    sentence = jiwer.RemoveWhiteSpace(replace_by_space=True)(sentence)
    # Remove multiple spaces.
    sentence = jiwer.RemoveMultipleSpaces()(sentence)
    # Remove white space in two end of string.
    sentence = jiwer.Strip()(sentence)

    # Convert all characters to upper.
    sentence = sentence.upper()

    return sentence


def calculate_measures(groundtruth, transcription):
    """Calculate character/word measures (hits, subs, inserts, deletes) for one given sentence"""
    groundtruth = normalize_sentence(groundtruth)
    transcription = normalize_sentence(transcription)

    # cer = ed.eval(transcription, groundtruth) / len(groundtruth)
    # c_result = jiwer.compute_measures([c for c in groundtruth if c != " "], [c for c in transcription if c != " "])
    c_result = jiwer.cer(groundtruth, transcription, return_dict=True)
    w_result = jiwer.compute_measures(groundtruth, transcription)

    return c_result, w_result, groundtruth, transcription

def transcribe(model, device, wav):
    """Calculate score on one single waveform"""
    # preparation
    inputs = model['processor'](wav, sampling_rate=16_000, return_tensors="pt", padding=True)
    
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.attention_mask.to(device)

    # forward
    with torch.no_grad():
        logits = model["model"](input_values, attention_mask=attention_mask).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    # transcription = model["tokenizer"].batch_decode(predicted_ids)[0]
    transcription = model['processor'].batch_decode(predicted_ids)[0]
    print(transcription)
    return transcription


# test_dataset = test_dataset.map(speech_file_to_array_fn)

# inputs = processor(test_dataset["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)

# with torch.no_grad():
#     logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

# predicted_ids = torch.argmax(logits, dim=-1)
# predicted_sentences = processor.batch_decode(predicted_ids)


# for i, predicted_sentence in enumerate(predicted_sentences):
#     import pdb
#     pdb.set_trace()
#     print("-" * 100)
#     print("Reference:", test_dataset[i]["sentence"])
#     print("Prediction:", predicted_sentence)


# # ASR_PRETRAINED_MODEL = "facebook/wav2vec2-large-960h-lv60-self"


# def load_asr_model(device):
#     """Load model"""
#     print(f"[INFO]: Load the pre-trained ASR by {ASR_PRETRAINED_MODEL}.")
#     model = Wav2Vec2ForCTC.from_pretrained(ASR_PRETRAINED_MODEL).to(device)
#     tokenizer = Wav2Vec2Tokenizer.from_pretrained(ASR_PRETRAINED_MODEL)
#     models = {"model": model, "tokenizer": tokenizer}
#     return models


# def normalize_sentence(sentence):
#     """Normalize sentence"""
#     # Convert all characters to upper.
#     sentence = sentence.upper()
#     # Delete punctuations.
#     sentence = jiwer.RemovePunctuation()(sentence)
#     # Remove \n, \t, \r, \x0c.
#     sentence = jiwer.RemoveWhiteSpace(replace_by_space=True)(sentence)
#     # Remove multiple spaces.
#     sentence = jiwer.RemoveMultipleSpaces()(sentence)
#     # Remove white space in two end of string.
#     sentence = jiwer.Strip()(sentence)

#     # Convert all characters to upper.
#     sentence = sentence.upper()

#     return sentence


# def calculate_measures(groundtruth, transcription):
#     """Calculate character/word measures (hits, subs, inserts, deletes) for one given sentence"""
#     groundtruth = normalize_sentence(groundtruth)
#     transcription = normalize_sentence(transcription)

#     # cer = ed.eval(transcription, groundtruth) / len(groundtruth)
#     # c_result = jiwer.compute_measures([c for c in groundtruth if c != " "], [c for c in transcription if c != " "])
#     c_result = jiwer.cer(groundtruth, transcription, return_dict=True)
#     w_result = jiwer.compute_measures(groundtruth, transcription)

#     return c_result, w_result, groundtruth, transcription


# def transcribe(model, device, wav):
#     """Calculate score on one single waveform"""
#     # preparation
#     inputs = model["tokenizer"](
#         wav, sampling_rate=16000, return_tensors="pt", padding="longest"
#     )
    
#     input_values = inputs.input_values.to(device)
#     attention_mask = inputs.attention_mask.to(device)

#     # forward
#     logits = model["model"](input_values, attention_mask=attention_mask).logits
#     predicted_ids = torch.argmax(logits, dim=-1)
#     transcription = model["tokenizer"].batch_decode(predicted_ids)[0]

#     return transcription
