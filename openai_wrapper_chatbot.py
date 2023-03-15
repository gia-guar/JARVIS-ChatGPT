import openai
import os
import whisper
import pyaudio
import sys
from chatgptwrapper.chatgpt_wrapper import ChatGPT
from ibm_watson import TextToSpeechV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from get_audio import *
import pygame

import JARVIS

# Set up the model
model_engine = "text-davinci-003"
prompt = "Hello, how are you today?"

## 1. AUTH
url ='your-watson-service-url'
apikey = 'your-ibm-cloud-api'

# Setup Service
authenticator = IAMAuthenticator(apikey)
# New tts service
tts = TextToSpeechV1(authenticator=authenticator)
# set serive url
tts.set_service_url(url)

# function to get ansers from ChatGPT
def generate_single_response(prompt, model_engine="text-davinci-003", temp=0.5):
    openai.api_key = 'your-openai-api-key'  
    prompt = (f"{prompt}")
    completions = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=1024,
        temperature=temp
    )
    return completions.choices[0].text

# function to get ansers from ChatGPT wrapper
def chatGPTwrap(bot,prompt, limit=100):
    prompt = prompt+'. answer with'+ str(limit)+' words or less'
    response = bot.ask(prompt)
    return bot, response 

## function to speak
def say(text,VoiceIdx='jarvis',Jarvis_voice=None):
    
    ## Pick the right voice
    voices = {'en':'en-US_AllisonV3Voice','it':'it-IT_FrancescaV3Voice'}

    ## CONVERT A STRING
    if VoiceIdx !='jarvis':
        with open('./answers/speech.mp3','wb') as audio_file:
            res = tts.synthesize(text, accept='audio/mp3', voice=voices[VoiceIdx]).get_result()
            audio_file.write(res.content)
    
        ##3 SAY OUTLOUD
        if pygame.mixer.get_init() is None:
            pygame.mixer.init()

        pygame.mixer.music.load("./answers/speech.mp3")
        pygame.mixer.music.set_volume(0.5)
        pygame.mixer.music.play()
        while(pygame.mixer.music.get_busy()): pass
        pygame.mixer.music.load(os.path.join('voices','empty.mp3'))
        os.remove("./answers/speech.mp3")
    else:
        JARVIS_voice.synthesize(text)
        JARVIS_voice.vocode()


### MAIN
if __name__=="__main__":
    bot = ChatGPT()
    whisper_model = whisper.load_model("medium")
    pygame.mixer.init()
    
    ## 1. AUTH
    url ='your-watson-serivc-url'
    apikey = 'your-api'
    # Setup Service
    authenticator = IAMAuthenticator(apikey)
    # New tts service
    tts = TextToSpeechV1(authenticator=authenticator)
    # set serive url
    tts.set_service_url(url)

    # INITIATE JARVIS
    JARVIS_voice = JARVIS.init_jarvis()


    while True:
        ## MIC TO TEXT
        record_to_file('output.wav')
        question,VoiceIdx = whisper_wav_to_text('output.wav',whisper_model)
        # check exit command
        if "THANKS" in question.upper():
            print('closing chat...')
            break
        #if "JARVIS" in question.upper():
        #    question = question.upper().strip('JARVIS')
        #    question = question.lower()
        #    VoiceIdx = 'jarvis'
        VoiceIdx = 'jarvis'
        
        ## TEXT >> CHAT RESPONSE
        print('ChatGPT(w):',end=' ')
        bot, response = chatGPTwrap(bot,question,limit=50)
        if "usable"in response.lower() and "chatgpt" in response.lower():
            print('wrapper unavailabe\nAttempting single prompt...')
            response = generate_single_response(question)
        print(response)

        ## RESPONSE >> SPOKEN RESPONSE
        response = response.replace(".",".\n")
        say(response,VoiceIdx,JARVIS_voice)

    say(". You're welcome. I'm glad I could help. bye",'jarvis',JARVIS_voice)
    