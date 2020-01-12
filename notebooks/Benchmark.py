#!/usr/bin/env python
# coding: utf-8

# This is to test TTS models with benchmark sentences for speech synthesis.
# 
# Before running this script please DON'T FORGET: 
# - to set file paths.
# - to download related model files from TTS and WaveRNN.
# - to checkout right commit versions (given next to the model) of TTS and WaveRNN.
# - to set the right paths in the cell below.
# 
# Repositories:
# - TTS: https://github.com/mozilla/TTS
# - WaveRNN: https://github.com/erogol/WaveRNN



TTS_PATH = "/home/chenghao03/tts"
WAVERNN_PATH ="/home/erogol/projects/"

import os
import sys
import io
import torch 
import time
import json
import numpy as np
from collections import OrderedDict
from pypinyin import lazy_pinyin, Style
style = Style.TONE3

# add libraries into environment
sys.path.append(TTS_PATH) # set this if TTS is not installed globally
sys.path.append(WAVERNN_PATH) # set this if TTS is not installed globally

import librosa
import librosa.display

from TTS.models.tacotron import Tacotron 
from TTS.layers import *
from TTS.utils.data import *
from TTS.utils.audio import AudioProcessor
from TTS.utils.generic_utils import load_config, setup_model
from TTS.utils.text import text_to_sequence
from TTS.utils.synthesis import synthesis
from TTS.utils.visual import visualize

# import IPython
# from IPython.display import Audio

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'


def tts(model, raw_text, CONFIG, use_cuda, ap, use_gl, figures=False, use_pinyin=False):
    if use_pinyin:
        text = " ".join(lazy_pinyin(raw_text, style=style))
    else:
        text = raw_text
    t_1 = time.time()
    waveform, alignment, mel_spec, mel_postnet_spec, stop_tokens = synthesis(model, text, CONFIG, use_cuda, ap, speaker_id, None, False)
    if CONFIG.model == "Tacotron" and not use_gl:
        # coorect the normalization differences b/w TTS and the Vocoder.
        mel_postnet_spec = ap.out_linear_to_mel(mel_postnet_spec.T).T
    mel_postnet_spec = ap._denormalize(mel_postnet_spec)
    if not use_gl:
        mel_postnet_spec = ap_vocoder._normalize(mel_postnet_spec)
        waveform = wavernn.generate(torch.FloatTensor(mel_postnet_spec.T).unsqueeze(0).cuda(), batched=batched_wavernn, target=8000, overlap=400)

    print(" >  Run-time: {}".format(time.time() - t_1))
    if figures:                                                                                                         
        visualize(alignment, mel_postnet_spec, stop_tokens, raw_text, ap.hop_length, CONFIG, mel_spec)                                                                       
    # IPython.display.display(Audio(waveform, rate=CONFIG.audio['sample_rate']))  
    os.makedirs(OUT_FOLDER, exist_ok=True)
    file_name = raw_text.replace(" ", "_").replace(".","") + f"-{speaker_id}.wav"
    out_path = os.path.join(OUT_FOLDER, file_name)
    ap.save_wav(waveform, out_path)
    return alignment, mel_postnet_spec, stop_tokens, waveform



# Set constants
# ROOT_PATH = '/home/chenghao03/tts/keep/aidatatang200zh-January-11-2020_02+36AM-3f83393'
ROOT_PATH = '/data/experiments/TTS/aidatatang200zh-January-12-2020_06+58PM-3f83393'
use_pinyin = True

# MODEL_NAME = "best_model.pth.tar"
MODEL_NAME = "checkpoint_30000.pth.tar"
MODEL_PATH = ROOT_PATH + '/' + MODEL_NAME
CONFIG_PATH = ROOT_PATH + '/config.json'
OUT_FOLDER = ROOT_PATH + '/benchmark_samples/' + MODEL_NAME
CONFIG = load_config(CONFIG_PATH)
# VOCODER_MODEL_PATH = "/media/erogol/data_ssd/Models/wavernn/ljspeech/mold_ljspeech_best_model/checkpoint_433000.pth.tar"
# VOCODER_CONFIG_PATH = "/media/erogol/data_ssd/Models/wavernn/ljspeech/mold_ljspeech_best_model/config.json"
# VOCODER_CONFIG = load_config(VOCODER_CONFIG_PATH)
use_cuda = True

# Set some config fields manually for testing
# CONFIG.windowing = False
# CONFIG.prenet_dropout = False
# CONFIG.separate_stopnet = True
CONFIG.use_forward_attn = True
# CONFIG.forward_attn_mask = True
# CONFIG.stopnet = True

# Set the vocoder
use_gl = True # use GL if True
batched_wavernn = True    # use batched wavernn inference if True


# LOAD TTS MODEL
from TTS.utils.text.symbols import symbols, phonemes

# multi speaker 
if CONFIG.use_speaker_embedding:
    speakers = json.load(open(f"{ROOT_PATH}/speakers.json", 'r'))
    speakers_idx_to_id = {v: k for k, v in speakers.items()}
else:
    speakers = []
    speaker_id = None

# load the model
num_chars = len(phonemes) if CONFIG.use_phonemes else len(symbols)
print(num_chars, len(phonemes), len(symbols))
model = setup_model(num_chars, len(speakers), CONFIG)

# load the audio processor
ap = AudioProcessor(**CONFIG.audio)         


# load model state
if use_cuda:
    cp = torch.load(MODEL_PATH)
else:
    cp = torch.load(MODEL_PATH, map_location=lambda storage, loc: storage)
print(cp.keys())
# load the model
model.load_state_dict(cp['model'])
if use_cuda:
    model.cuda()
model.eval()
print(cp['step'])
print(cp['r'])

# set model stepsize
if 'r' in cp:
    model.decoder.set_r(cp['r'])



# LOAD WAVERNN
if use_gl == False:
    from WaveRNN.models.wavernn import Model
    from WaveRNN.utils.audio import AudioProcessor as AudioProcessorVocoder
    bits = 10
    ap_vocoder = AudioProcessorVocoder(**VOCODER_CONFIG.audio)    
    wavernn = Model(
            rnn_dims=512,
            fc_dims=512,
            mode=VOCODER_CONFIG.mode,
            mulaw=VOCODER_CONFIG.mulaw,
            pad=VOCODER_CONFIG.pad,
            upsample_factors=VOCODER_CONFIG.upsample_factors,
            feat_dims=VOCODER_CONFIG.audio["num_mels"],
            compute_dims=128,
            res_out_dims=128,
            res_blocks=10,
            hop_length=ap_vocoder.hop_length,
            sample_rate=ap_vocoder.sample_rate,
            use_upsample_net = True,
            use_aux_net = True
        ).cuda()

    check = torch.load(VOCODER_MODEL_PATH)
    wavernn.load_state_dict(check['model'], strict=False)
    if use_cuda:
        wavernn.cuda()
    wavernn.eval();
    print(check['step'])


# ### Comparision with https://mycroft.ai/blog/available-voices/

model.eval()
model.decoder.max_decoder_steps = 2000
# speaker_id = 0
for speaker_id in range(5):
    sentence =  "怎么网络不好啊,为什么上不去"
    align, spec, stop_tokens, wav = tts(model, sentence, CONFIG, use_cuda, ap, use_gl=use_gl, figures=False, use_pinyin=use_pinyin)
        
    sentence =  "晚饭吃什么"
    align, spec, stop_tokens, wav = tts(model, sentence, CONFIG, use_cuda, ap, use_gl=use_gl, figures=False, use_pinyin=use_pinyin)
    
    sentence =  "若一扇门长和宽各是几厘米才是黄金分割零点六幺八"
    align, spec, stop_tokens, wav = tts(model, sentence, CONFIG, use_cuda, ap, use_gl=use_gl, figures=False, use_pinyin=use_pinyin)
   
    # sentence = "海南省将于何时决定是否进行高中文理分科"  
    # align, spec, stop_tokens, wav = tts(model, sentence, CONFIG, use_cuda, ap, use_gl=use_gl, figures=False, use_pinyin=use_pinyin)
