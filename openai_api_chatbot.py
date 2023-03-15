print('importing dependancies...')
import openai
import os
import whisper
import pyaudio
import sys
import pygame
from datetime import datetime
import copy

from ibm_watson import TextToSpeechV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

import get_audio as myaudio
import JARVIS

print('DONE')

openai.api_key = 'your OpenAI API Key'

# Set up the model
model_engine = "text-davinci-003"
prompt = "Hello, how are you today?"

# function to get single answer from ChatGPT
def generate_single_response(prompt, model_engine="text-davinci-003", temp=0.5):
    openai.api_key = 'your OpenAI API Key'  
    prompt = (f"{prompt}")
    completions = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=1024,
        temperature=temp
    )
    return completions.choices[0].text

# [NEW!] function to get answer from official openai API
def send_API_request(chat_history):
    API_response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=chat_history)

    answer = API_response['choices'][0]['message']['content']

    chat_history.append({"role": "assistant", "content":answer})

    return chat_history


## function to speak
def say(text, VoiceIdx='jarvis',JARVIS_voice=None):
    
    ## Pick the right voice
    voices = {'en':'en-US_AllisonV3Voice','it':'it-IT_FrancescaV3Voice'}

    ## CONVERT A STRING
    if VoiceIdx !='jarvis':
        if not os.path.isdir('saved_chats'): os.mkdir("saved_chats")
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

# [new!] function to save chat:
def save_chat(history):
    if not os.path.isdir('saved_chats'): os.mkdir("saved_chats")
    
    temp = copy.deepcopy(history)
    temp.append({"role":"user", "content":"generate a title for this conversation"})
    temp = send_API_request(temp)
    title = temp[-1]["content"]

    title = title.replace(' ','_')
    title = ''.join(e for e in title if e.isalnum())

    fname = str( str(datetime.today().strftime('%Y-%m-%d')) + '_' + str(title))
    with open(os.path.join('saved_chats', fname), 'w') as f:
        for message in history:
            f.write(message["role"]+ ': ' + message["content"]+'\n')
        f.close()

### MAIN
if __name__=="__main__":
    
    print('loading whisper model...')
    whisper_model = whisper.load_model("medium")

    print('opening pygame: ')
    pygame.mixer.init()
    
    ## 1. AUTH
    print('Authorizing IBM Cloud...')
    url ='your IMB CLOUD tts service url'
    apikey = 'your IBM CLOUD API Key'
    # Setup Service
    print('Setting up cloud authenticator...')
    authenticator = IAMAuthenticator(apikey)
    # New tts service
    print('Setting up text-to-speech...')
    tts = TextToSpeechV1(authenticator=authenticator)
    # set serive url
    print('Setting up cloud service ...')
    tts.set_service_url(url)

    print('DONE\n\n')
    # INITIATE JARVIS

    print('initiating JARVIS voice...')
    JARVIS_voice = JARVIS.init_jarvis()

    chat_history = [{"role": "system", "content": "You are a helpful assistant."}]

    while True:

        ## MIC TO TEXT
        print('setting up microphone...')
        myaudio.record_to_file('output.wav')
        question, detected_language = myaudio.whisper_wav_to_text('output.wav', whisper_model)
        
        # check exit command
        if "THANKS" in question.upper():
            print('closing chat...')
            break
        
        if "HEY JARVIS" in question.upper() and detected_language=='en':
            question = question.upper().replace('HEY JARVIS', '')
            question = question.lower()
            VoiceIdx = 'jarvis'
        else:
            VoiceIdx = detected_language

        chat_history.append({"role":"user", "content":question})
        
        ## TEXT >> CHAT RESPONSE
        chat_history = send_API_request(chat_history)
        response = chat_history[-1]["content"]

        ## RESPONSE >> SPOKEN RESPONSE
        response = response.replace(".",".\n")
        say(response,VoiceIdx,JARVIS_voice)

        print('\n')

    save_chat(chat_history)
    say(". You're welcome. I'm glad I could help. bye", 'en')