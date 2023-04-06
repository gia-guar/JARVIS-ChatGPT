print('### IMPORTING DEPENDANCIES ###')
import os
import whisper
import pygame

from Assistant import get_audio as myaudio
from Assistant.VirtualAssistant import VirtualAssistant
from Assistant.tools import count_tokens

os.environ['OPENAI_API_KEY']  = 'your-openai-api-key'
os.environ['IBM_API_KEY']     = 'your-ibm-cloud-api-key'
os.environ['IBM_TTS_SERVICE'] = 'your-ibm-cloud-tts-url'

print('DONE\n')


### MAIN
if __name__=="__main__":
    print("### SETTING UP ENVIROMENT ###")
    
    SOUND_DIR = os.path.join('sounds')
    
    print('loading whisper model...')
    whisper_model = whisper.load_model("large") # pick the one that works best for you, but remember: only medium and large are multi language

    print('opening pygame:')
    pygame.mixer.init()
    
    # INITIATE JARVIS
    print('initiating JARVIS voice...')
    jarvis = VirtualAssistant(
        openai_api = os.getenv('OPENAI_API_KEY'),
        ibm_api    = os.getenv('IBM_API_KEY'),
        ibm_url    = os.getenv('IBM_TTS_SERVICE'),
        voice_id   = 'friday_en',
        whisper_model= whisper_model,
        awake_with_keywords=["elephant"],
        model= "gpt-3.5-turbo",
        embed_model= "text-embedding-ada-002",
        RESPONSE_TIME = 3,
        SLEEP_DELAY = 30,
        )

    while True:

        if not(jarvis.is_awake):
            print('\n awaiting for triggering words...')

            while not(jarvis.is_awake):
                jarvis.listen_passively()
        
        jarvis.record_to_file('output.wav')
        

        if jarvis.is_awake:
            question, detected_language = myaudio.whisper_wav_to_text('output.wav', whisper_model, prior=jarvis.languages.keys())

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
            flag = jarvis.analyze_prompt()

            if "1" in flag or "find a file" in flag:
                summary = jarvis.find_file()
                print(summary)
                print(jarvis.current_conversation)
                continue
            
            # count tokens to satisfy the max limits
            # <to do>
            response = jarvis.get_answer(question, update=False)
            jarvis.say(response, VoiceIdx=VoiceIdx)

            print('\n')
