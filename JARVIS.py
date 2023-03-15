import time
from encoder.inference import plot_embedding_as_heatmap
from toolbox.utterance import Utterance
from pathlib import Path
from typing import List, Set
import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np
from time import sleep
import umap
import sys
from warnings import filterwarnings
filterwarnings("ignore")

from toolbox.ui import UI
from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder
from pathlib import Path
from time import perf_counter as timer
from toolbox.utterance import Utterance
import numpy as np
import traceback
import sys
import os

recognized_datasets = [
    "LibriSpeech/dev-clean",
    "LibriSpeech/dev-other",
    "LibriSpeech/test-clean",
    "LibriSpeech/test-other",
    "LibriSpeech/train-clean-100",
    "LibriSpeech/train-clean-360",
    "LibriSpeech/train-other-500",
    "LibriTTS/dev-clean",
    "LibriTTS/dev-other",
    "LibriTTS/test-clean",
    "LibriTTS/test-other",
    "LibriTTS/train-clean-100",
    "LibriTTS/train-clean-360",
    "LibriTTS/train-other-500",
    "LJSpeech-1.1",
    "VoxCeleb1/wav",
    "VoxCeleb1/test_wav",
    "VoxCeleb2/dev/aac",
    "VoxCeleb2/test/aac",
    "VCTK-Corpus/wav48",
]

class VoiceCloner:
    def __init__(self,voice_dir,enc_dir,voc_dir,syn_dir):
        self.speaker_name='JARVIS'
        self.voice_dir=voice_dir
        self.enc_dir = enc_dir
        self.voc_dir = voc_dir
        self.syn_dir = syn_dir
        self.synthesizer = None
        self.current_generated = (None, None, None, None) # speaker_name, spec, breaks, wav
        self.utterances = set()
        self.init_encoder()
        self.init_synthesizer()
        self.init_vocoder()

    def play(self, wav, sample_rate):
            sd.stop()
            sd.play(wav, sample_rate)
            sd.wait()

    def load_from_browser(self, fpath):
        path = Path(fpath)
        speaker_name = path.parent.name
        name = path.name
        wav = Synthesizer.load_preprocess_wav(fpath)
        
        self.add_real_utterance(wav, name, speaker_name)
    
    def add_real_utterance(self,wav,name,speaker_name):
        # Compute the mel spectrogram
        spec = Synthesizer.make_spectrogram(wav)

        # Compute the embedding
        if not encoder.is_loaded():
            print('encoder not loaded... initiating encoder')
            self.init_encoder()

        encoder_wav = encoder.preprocess_wav(wav)
        embed, partial_embeds, _ = encoder.embed_utterance(encoder_wav, return_partials=True)
        utterance = Utterance(name, speaker_name, wav, spec, embed, partial_embeds, False)
        self.utterances.add(utterance)

    def init_encoder(self):
        encoder.load_model(Path(self.enc_dir))

    def init_vocoder(self):
        vocoder.load_model(self.voc_dir)
    
    def init_synthesizer(self):
        self.synthesizer = Synthesizer(self.syn_dir)

    def synthesize(self, text):
        texts=text.split('\n')
        if self.synthesizer is None:
            model_dir = Path('saved_models\default')
            checkpoints_dir = model_dir.joinpath("taco_pretrained")
            self.synthesizer = Synthesizer(checkpoints_dir)
            
        with open(os.path.join('saved_models','embeds','embed_9.npy'), 'rb') as f:
            embed = np.load(f)
            
        embeds = [embed] * len(texts)
        specs = self.synthesizer.synthesize_spectrograms(texts, embeds)
        breaks = [spec.shape[1] for spec in specs]
        spec = np.concatenate(specs, axis=1)
        self.current_generated = (self.speaker_name, spec, breaks, None)

    def vocode(self):
        speaker_name, spec, breaks, _ = self.current_generated
        wav = vocoder.infer_waveform(spec)
        # add breaks
        b_ends = np.cumsum(np.array(breaks) * Synthesizer.hparams.hop_size)
        b_starts = np.concatenate(([0], b_ends[:-1]))
        wavs = [wav[start:end] for start, end, in zip(b_starts, b_ends)]
        breaks = [np.zeros(int(0.15 * Synthesizer.sample_rate))] * len(breaks)
        wav = np.concatenate([i for w, b in zip(wavs, breaks) for i in (w, b)])

        # Play it
        wav = wav / np.abs(wav).max() * 0.97
        print(Synthesizer.sample_rate)
        self.play(wav, Synthesizer.sample_rate)
        wav_name = speaker_name + "_gen_%05d" % np.random.randint(100000)





def init_jarvis():
    JARVIS = VoiceCloner(
        voice_dir=os.path.join('voices'),
        enc_dir=Path(os.path.join('saved_models','default','encoder.pt')),
        voc_dir=Path(os.path.join('saved_models','default','vocoder.pt')),
        syn_dir=Path(os.path.join('saved_models','default','synthesizer.pt'))
    )
    return JARVIS