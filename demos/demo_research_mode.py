print('### LOADING CREDENTIALS ###')
from dotenv import load_dotenv
import os

from Assistant.research_mode import ResearchAssistant
from Assistant.semantic_scholar.agent_tools import readPDF, update_workspace_dataframe

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
import pinecone
print('DONE\n')

### MAIN
if __name__=="__main__":
    print("### SETTING UP ENVIROMENT ###")
    OFFLINE = False
    pygame.mixer.init()
    # INITIATE JARVIS
    print('initiating JARVIS voice...')
    jarvis = VirtualAssistant(
        openai_api   = os.getenv('OPENAI_API_KEY'),
        ibm_api      = os.getenv('IBM_API_KEY'),
        ibm_url      = os.getenv('IBM_TTS_SERVICE'),
        elevenlabs_api = os.getenv('ELEVENLABS_API_KEY'),
        elevenlabs_voice = 'Antoni',
        voice_id     = {'en':'jarvis_en'},
        whisper_size = 'medium',
        awake_with_keywords=["jarvis"],
        model= "gpt-3.5-turbo",
        embed_model= "text-embedding-ada-002",
        RESPONSE_TIME = 3,
        SLEEP_DELAY = 30,
        mode = 'RESEARCH',
        )

    jarvis.init_research_mode()
    i = 0
    while True:  
        prompt = input("user: ")
        # check exit command
        if "THANKS" in prompt.upper() or len(prompt.split())<=1:
            jarvis.go_to_sleep()
            continue
        
        jarvis.expand_conversation(role="user", content=prompt)

        # PROMPT MANAGING [BETA]
        flag = jarvis.analyze_prompt(prompt)

        print(flag)
        # redirect the conversation to an action manager or to the LLM
        if "1" in flag or "tool" in flag:
            print('(thought): action')
            response = jarvis.use_tools(prompt)
            response = response
        
        elif "2" in flag or "respond" in flag:
            print('(thought): response')
            response = jarvis.get_answer(prompt)
        else:
            print(f'(thought): {flag}: workspace')
            input('> continue?')
            response = jarvis.secondary_agent(prompt)
            

        jarvis.expand_conversation(role='assistant', content=response)
        pygame.mixer.stop()
        jarvis.say(response, VoiceIdx='en', elevenlabs=True)

        i+=1