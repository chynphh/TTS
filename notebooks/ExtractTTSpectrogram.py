#!/usr/bin/env python
# coding: utf-8

# This is a notebook to generate mel-spectrograms from a TTS model to be used for WaveRNN training.



TTS_PATH = "/home/chenghao03/tts"
import os
import sys
sys.path.append(TTS_PATH)
import torch
import importlib
import numpy as np
from tqdm import tqdm as tqdm
from torch.utils.data import DataLoader
from TTS.models.tacotron2 import Tacotron2
from TTS.datasets.TTSDataset import MyDataset
from TTS.utils.audio import AudioProcessor
from TTS.utils.visual import plot_spectrogram
from TTS.utils.generic_utils import load_config, setup_model
from TTS.datasets.preprocess import ljspeech
import os
from TTS.utils.speakers import load_speaker_mapping
os.environ['CUDA_VISIBLE_DEVICES']='0'




def set_filename(wav_path, out_path):
    wav_file = os.path.basename(wav_path)
    file_name = wav_file.split('.')[0]
    os.makedirs(os.path.join(out_path, "quant"), exist_ok=True)
    os.makedirs(os.path.join(out_path, "mel"), exist_ok=True)
    os.makedirs(os.path.join(out_path, "wav_gl"), exist_ok=True)
    wavq_path = os.path.join(out_path, "quant", file_name)
    mel_path = os.path.join(out_path, "mel", file_name)
    wav_path = os.path.join(out_path, "wav_gl", file_name)
    return file_name, wavq_path, mel_path, wav_path


def setup_loader(c, ap, meta_file):
    dataset = MyDataset(
        c.r,
        c.text_cleaner,
        meta_data=meta_file,
        ap=ap,
        batch_group_size=c.batch_group_size * c.batch_size,
        min_seq_len=c.min_seq_len,
        max_seq_len=c.max_seq_len,
        phoneme_cache_path=c.phoneme_cache_path,
        use_phonemes=c.use_phonemes,
        phoneme_language=c.phoneme_language,
        enable_eos_bos=c.enable_eos_bos_chars,
        verbose=False)
    loader = DataLoader(
        dataset,
        batch_size= c.batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        drop_last=False,
        sampler=None,
        num_workers=c.num_loader_workers,
        pin_memory=False)
    return loader


# def get_num_of_speaker(C)
#     if c.use_speaker_embedding:
#         speakers = get_speakers(meta_data_train)
#         if args.restore_path:
#             prev_out_path = os.path.dirname(args.restore_path)
#             speaker_mapping = load_speaker_mapping(prev_out_path)
#             assert all([speaker in speaker_mapping
#                         for speaker in speakers]), "As of now you, you cannot " \
#                                                    "introduce new speakers to " \
#                                                    "a previously trained model."
#         num_speakers = len(speaker_mapping)
#     else:
#         num_speakers = 0
#     return num_speakers

OUT_PATH = "/data/aidatatang_200zh/wavernn/1/checkpoint_90000"
DATA_PATH = "/data/aidatatang_200zh/corpus"
DATASET = "aidatatang_tts2"
METADATA_FILE = None
TTS_MODLE_PATH = "/home/chenghao03/tts/keep/aidatatang200zh-January-11-2020_02+36AM-3f83393/"
CONFIG_PATH = TTS_MODLE_PATH + "config.json"
MODEL_FILE = TTS_MODLE_PATH + "checkpoint_90000.pth.tar"
DRY_RUN = False   # if False, does not generate output files, only computes loss and visuals.
BATCH_SIZE = 32

use_cuda = torch.cuda.is_available()
print(" > CUDA enabled: ", use_cuda)

C = load_config(CONFIG_PATH)
ap = AudioProcessor(bits=9, **C.audio)
print(ap)
C.prenet_dropout = False
C.separate_stopnet = True




preprocessor = importlib.import_module('TTS.datasets.preprocess')
preprocessor = getattr(preprocessor, DATASET.lower())

meta_data = preprocessor(DATA_PATH)
loader = setup_loader(C, ap, meta_data)



from TTS.utils.text.symbols import symbols, phonemes
from TTS.utils.generic_utils import sequence_mask
from TTS.layers.losses import L1LossMasked
from TTS.utils.text.symbols import symbols, phonemes

# load the model
num_chars = len(phonemes) if C.use_phonemes else len(symbols)
speaker_mapping = load_speaker_mapping(TTS_MODLE_PATH)
num_speakers = len(speaker_mapping)

model = setup_model(num_chars, num_speakers, C)
checkpoint = torch.load(MODEL_FILE)
model.load_state_dict(checkpoint['model'])
print(checkpoint['step'])
model.eval()
if use_cuda:
    model = model.cuda()


# ### Generate model outputs 
import pickle

file_idxs = []
losses = []
postnet_losses = []
criterion = L1LossMasked()
for data in tqdm(loader):
    # setup input data
    text_input = data[0]
    text_lengths = data[1]
    speaker_names = data[2]
    linear_input = data[3] if C.model in ["Tacotron", "TacotronGST"
                                            ] else None
    mel_input = data[4]
    mel_lengths = data[5]
    stop_targets = data[6]
    item_idx = data[7]

    if C.use_speaker_embedding:
        speaker_ids = [
            speaker_mapping[speaker_name] for speaker_name in speaker_names
        ]
        speaker_ids = torch.LongTensor(speaker_ids)
    else:
        speaker_ids = None

        # set stop targets view, we predict a single stop token per r frames prediction
        stop_targets = stop_targets.view(text_input.shape[0],
                                         stop_targets.size(1) // c.r, -1)
        stop_targets = (stop_targets.sum(2) >
                        0.0).unsqueeze(2).float().squeeze(2)

    # dispatch data to GPU
    if use_cuda:
        text_input = text_input.cuda()
        text_lengths = text_lengths.cuda()
        mel_input = mel_input.cuda()
        mel_lengths = mel_lengths.cuda()
        if linear_input is not None:
            linear_input = linear_input.cuda()
        stop_targets = stop_targets.cuda()
        if speaker_ids is not None:
            speaker_ids = speaker_ids.cuda()

    mask = sequence_mask(text_lengths)
    # print(text_input, text_lengths, mel_input, speaker_ids)
    mel_outputs, postnet_outputs, alignments, stop_tokens = model(text_input, text_lengths, mel_input, speaker_ids=speaker_ids)
    # print(mel_outputs, postnet_outputs, alignments, stop_tokens)
    # compute mel specs from linear spec if model is Tacotron
    mel_specs = []
    if C.model == "Tacotron":
        postnet_outputs = postnet_outputs.data.cpu().numpy()
        for b in range(postnet_outputs.shape[0]):
            postnet_output = postnet_outputs[b]
            mel_specs.append(torch.FloatTensor(ap.out_linear_to_mel(postnet_output.T).T).cuda())
        postnet_outputs = torch.stack(mel_specs)
    
    loss = criterion(mel_outputs, mel_input, mel_lengths)
    loss_postnet = criterion(postnet_outputs, mel_input, mel_lengths)
    losses.append(loss.item())
    postnet_losses.append(loss_postnet.item())
    if not DRY_RUN:
        for idx in range(text_input.shape[0]):
            wav_file_path = item_idx[idx]
            file_name, wavq_path, mel_path, wav_path = set_filename(wav_file_path, OUT_PATH)
            # print(file_name)
            file_idxs.append(file_name)

            # # quantize and save wav
            # wav = ap.load_wav(wav_file_path)
            # wavq = ap.quantize(wav)
            # np.save(wavq_path, wavq)

            # save TTS mel
            mel = postnet_outputs[idx]
            mel = mel.data.cpu().numpy()
            mel_length = mel_lengths[idx]
            mel = mel[:mel_length, :].T
            np.save(mel_path, mel)

            # # save GL voice
            # wav_gen = ap.inv_mel_spectrogram(mel.T) # mel to wav
            # wav_gen = ap.quantize(wav_gen)
            # np.save(wav_path, wav_gen)

if not DRY_RUN:
    pickle.dump(file_idxs, open(OUT_PATH+"/dataset_ids.pkl", "wb"))      
    

print(np.mean(losses))
print(np.mean(postnet_losses))



# # ### Check model performance
# idx = 1
# mel_example = postnet_outputs[idx].data.cpu().numpy()
# plot_spectrogram(mel_example[:mel_lengths[idx], :], ap);
# print(mel_example[:mel_lengths[1], :].shape)




# mel_example = mel_outputs[idx].data.cpu().numpy()
# plot_spectrogram(mel_example[:mel_lengths[idx], :], ap);
# print(mel_example[:mel_lengths[1], :].shape)




# wav = ap.load_wav(item_idx[idx])
# melt = ap.melspectrogram(wav)
# print(melt.shape)
# plot_spectrogram(melt.T, ap);




# # postnet, decoder diff
# from matplotlib import pylab as plt
# mel_diff = mel_outputs[idx] - postnet_outputs[idx]
# plt.figure(figsize=(16, 10))
# plt.imshow(abs(mel_diff.detach().cpu().numpy()[:mel_lengths[idx],:]).T,aspect="auto", origin="lower");
# plt.colorbar()
# plt.tight_layout()




# from matplotlib import pylab as plt
# # mel = mel_poutputs[idx].detach().cpu().numpy()
# mel = postnet_outputs[idx].detach().cpu().numpy()
# mel_diff2 = melt.T - mel[:melt.shape[1]]
# plt.figure(figsize=(16, 10))
# plt.imshow(abs(mel_diff2).T,aspect="auto", origin="lower");
# plt.colorbar()
# plt.tight_layout()






