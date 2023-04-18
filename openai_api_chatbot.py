print('### LOADING CREDENTIALS ###')
from dotenv import load_dotenv
import os

load_dotenv()

if len(os.environ['OPENAI_API_KEY'])==0: 
    print('openai API key not detected in .env')
    raise Exception("[$] openai API key is required. Learn more at https://platform.openai.com/account/api-keys")

if len(os.environ['IBM_API_KEY'])==0: print('[free] IBM cloud API Key not detected in .env\nLearn more at: https://cloud.ibm.com/catalog/services/text-to-speech')

if len(os.environ['IBM_TTS_SERVICE'])==0: print('[free] IBM cloud TTS service not detected in .env\nLearn more at: https://cloud.ibm.com/catalog/services/text-to-speech')

use_porcupine = True
if len(os.environ['PORCUPINE_KEY']) == 0: 
    print('[free] PicoVoice not detected in .env\nLearn more at: https://picovoice.ai/platform/porcupine/')
    use_porcupine = False


print('DONE\n')

print('### IMPORTING DEPENDANCIES ###')
import pygame

from Assistant import get_audio as myaudio
from Assistant.VirtualAssistant import VirtualAssistant
from Assistant.tools import count_tokens


print('DONE\n')


### MAIN
if __name__=="__main__":
    print("### SETTING UP ENVIROMENT ###")
    OFFLINE = True
    SOUND_DIR = os.path.join('sounds')
    
    print('loading whisper model...')

    print('opening pygame:')
    pygame.mixer.init()
    
    # INITIATE JARVIS
    print('initiating JARVIS voice...')
    jarvis = VirtualAssistant(
        openai_api   = os.getenv('OPENAI_API_KEY'),
        ibm_api      = os.getenv('IBM_API_KEY'),
        ibm_url      = os.getenv('IBM_TTS_SERVICE'),
        elevenlabs_api = os.getenv('ELEVENLABS_API_KEY'),
        elevenlabs_voice = '',
        voice_id     = 'jarvis_en',
        whisper_size = 'medium',
        awake_with_keywords=["jarvis"],
        model= "gpt-3.5-turbo",
        offline_model= 'anon8231489123_vicuna-13b-GPTQ-4bit-128g', # Runs on GPU (lots of space required)
        embed_model= "text-embedding-ada-002",
        RESPONSE_TIME = 3,
        SLEEP_DELAY = 30,
        )

    while True:
        if not(jarvis.is_awake):
            print('\n awaiting for triggering words...')

            #block until the wakeword is heard, using porcupine
            if use_porcupine:
                jarvis.block_until_wakeword()
            else:
                while not(jarvis.is_awake):
                    jarvis.listen_passively()
        
        jarvis.record_to_file('output.wav')
        

        if jarvis.is_awake:
            question, detected_language = myaudio.whisper_wav_to_text('output.wav', jarvis.interpreter, prior=jarvis.languages.keys())

            # check exit command
            if "THANKS" in question.upper() or len(question.split())<=1:
                jarvis.go_to_sleep()
                continue
            
            if detected_language=='en':
                VoiceIdx = 'jarvis'
            else:
                VoiceIdx = detected_language
            
            # PROMPT MANAGING [BETA]
            jarvis.expand_conversation(role="user", content=question)

            flag = "2" 
            if not(OFFLINE):
                flag = jarvis.analyze_prompt()

            try:
                print('(thought): ', flag)
            except: pass
            if "1" in flag or "find a file" in flag:
                summary = jarvis.find_file()
                print(summary)
                continue
            
            # count tokens to satisfy the max limits
            # <to do>
            response = jarvis.get_answer(question, update=False, mode='offline'if OFFLINE else 'online')
            jarvis.say(response, VoiceIdx=VoiceIdx, IBM=True)

            print('\n')
