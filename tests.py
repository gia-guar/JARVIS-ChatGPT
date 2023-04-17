"""
RUN THIS TO CHECK FOR ERRORS AFTER INSTALLATION
It tries to do some stuff that might raise errors. If tracebacks occurs something might wrong with your environment
or APIs credentials
"""

import os
import whisper
import pygame
import Assistant.get_audio as myaudio
import Assistant.tools
import sys
import time
import openai
from contextlib import contextmanager 
from dotenv import load_dotenv

from Assistant import get_audio as myaudio
from Assistant.VirtualAssistant import VirtualAssistant
from Assistant.tools import count_tokens

def load_keys():
    load_dotenv()
    if len(os.environ['OPENAI_API_KEY'])==0: 
        print('openai API key not detected in .env')
        raise Exception("[$] openai API key is required. Learn more at https://platform.openai.com/account/api-keys")
    openai.api_key = os.environ['OPENAI_API_KEY']
    if len(os.environ['IBM_API_KEY'])==0: print('[free] IBM cloud API Key not detected in .env\nLearn more at: https://cloud.ibm.com/catalog/services/text-to-speech')
    if len(os.environ['IBM_TTS_SERVICE'])==0: print('[free] IBM cloud TTS service not detected in .env\nLearn more at: https://cloud.ibm.com/catalog/services/text-to-speech')
    if len(os.environ['PORCUPINE_KEY']) == 0: print('[free] PicoVoice not detected in .env\nLearn more at: https://picovoice.ai/platform/porcupine/')
    

def check_whisper():
    whisper_model = whisper.load_model("small")
    response, _ = myaudio.whisper_wav_to_text(os.path.join('Assistant', 'voices', 'jarvis_en.wav'), whisper_model)
    assert isinstance(response, str)

def check_openai():
    chat = [{"role": "system", "content": "You are a helpful assistant"},{"role":"user", "content":"say hi"}]
    now = time.perf_counter()
    openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0,
        max_tokens=10,
        messages=chat)
    
    then = time.perf_counter()
    print(f'time to get a simple answer from gpt-3.5-turbo: {then-now} seconds')

    now = time.perf_counter()
    openai.Embedding.create(input='yes', engine='text-embedding-ada-002')
    then = time.perf_counter()
    print(f'time to get a extract an Embedding with text-embedding-ada-002: {then-now} seconds')

    

def check_virtual_assistant():
    whisper_model = whisper.load_model("small")  
    VA = VirtualAssistant(openai_api = os.getenv('OPENAI_API_KEY'),
        ibm_api    = os.getenv('IBM_API_KEY'),
        ibm_url    = os.getenv('IBM_TTS_SERVICE'),
        voice_id   = 'jarvis_en',
        whisper_model= whisper_model,
        awake_with_keywords=["jarvis"],
        model= "gpt-3.5-turbo",
        embed_model= "text-embedding-ada-002",
        RESPONSE_TIME = 3,
        SLEEP_DELAY = 30,
    )
    
    now = time.perf_counter()
    VA.translator.translate('ciao questo testo di moderate dimensioni da tradurre in lingua inglese', from_language='it', to_language='en')
    then = time.perf_counter()
    print(f'time to get translate a short text: {then-now} seconds')

    VA.say('Welcome.\n I am Just A Very Intelligent System here to help', VoiceIdx='jarvis')
    VA.say('Welcome.\n I am Just A Very Intelligent System here to help', VoiceIdx='en')


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

if __name__=='__main__':

    load_keys()
    check_openai()
    with suppress_stdout():
        check_whisper()

    check_virtual_assistant()
