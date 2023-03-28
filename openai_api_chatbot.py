print('### IMPORTING DEPENDANCIES ###')
import openai
import os
import whisper
import pyaudio
import sys
import pygame
from datetime import datetime
import copy
import time
import pyttsx3

from ibm_watson import TextToSpeechV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

import get_audio as myaudio
import JARVIS

openai.api_key = 'your-openai-api-key'

print('DONE\n')

""" SECTION 1: CHATGPT FUNCTIONS
In this section you'll find all the function that handles openai API services
"""

# [deprecated] function to get single answer from ChatGPT 
def generate_single_response(prompt, model_engine="text-davinci-003", temp=0.5):
    # Set up the model
    model_engine = "text-davinci-003"
    prompt = "Hello, how are you today?"
    prompt = (f"{prompt}")
    completions = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=1024,
        temperature=temp
    )
    return completions.choices[0].text

# function to get answer from official openai API
def send_API_request(chat_history):
    API_response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=chat_history)

    answer = API_response['choices'][0]['message']['content']

    chat_history.append({"role": "assistant", "content":answer})

    return chat_history


""" SECTION 2: TEXT-TO-SPEECH
the 'say' funnction take in input some text and uses a text to speech (TTS) models to reproduce some sounds on the system speakers:
there are 3 possible TTS:

    - IBM Watson: it's a could service, it needs connection and (free) credit. It's fast, it sound nice and understandable. It can speak in any languages 
    you want 

    - pyttsx3: It's a python package. It doesn't sound as good as IBM but it's 100% free and it's fast+understandable.
    the pyttsx3 engine can speak in any language that is supported in your machine. In mine, i got installed Italian and British English, if I wanted
    portuguese I'd need to install a new system tts voice (https://www.youtube.com/watch?v=KMtLqPi2wiU&ab_channel=MuruganS).

    - JARVIS (Tacotron): it's a pretrained model that use an embedding to generate outputs. It's slower, the audio might sound glitchy and the text
    needs to be made of "short" (15-24 words) sentences. These are the optimal settings for a readable output.
    You can ask ChatGPT to make responses in this way (I made a system prompt specifying how to make the answers). 

    Maybe JARVIS is not the best but honestly is just too cool to have such option.
"""
def say(text, VoiceIdx='jarvis',JARVIS_voice=None):
    
    ## Pick the right voice
    voices = {'en':'en-US_AllisonV3Voice','it':'it-IT_FrancescaV3Voice'}

    ## CONVERT A STRING
    if VoiceIdx !='jarvis':
        if not os.path.isdir('saved_chats'): os.mkdir("saved_chats")
        with open(os.path.join(os.getcwd(), 'answers','speech.mp3'),'wb') as audio_file:
            try:
                res = tts.synthesize(text, accept='audio/mp3', voice=voices[VoiceIdx]).get_result()
                audio_file.write(res.content)
            except:
                print('*!* IBM credit likely ended *!*  > using pyttsx3 for voice generation)')
                print('\n[assistant]: '+text)
                engine = pyttsx3.init()
                engine = change_voice(engine, lang_id=VoiceIdx)
                engine.say(text)
                engine.runAndWait()
                return
    
        ## SAY OUTLOUD
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

# works with pyttsx3 text to speech engine
def change_voice(engine, lang_id):
    languages = {
        'it': "Italian",
        'en': "English",
        # add yours
    }
    for voice in engine.getProperty('voices'):
        if languages[lang_id] in voice.name:
            engine.setProperty('voice', voice.id)
            return engine
    
    raise Exception


""" SECTION 3: ASSISTANT [temp]
Here lies all the functions that handles the textual prompt. All these will be incorporated inside a ASSISTANT object in future
Functions to be added:
    - flag extraction: make a function that extract keywords or phrases and retrun a set of actions to be done;
    - action exectution: given some flags, perform OS actions like saving chats, accessing and writing other files, manage memory; 
    - Device interaction: perform Smart Home tasks like with Alexa/Siri;
"""

# [new!] function to save chat:
def save_chat(history):
    if not os.path.isdir('saved_chats'): os.mkdir("saved_chats")
    
    temp = copy.deepcopy(history)
    temp.append({"role":"user", "content":"generate a title for this conversation"})
    temp = send_API_request(temp)
    title = temp[-1]["content"]

    say(f'I am saving this conversation with title: {title}', VoiceIdx='en')

    title = title.replace(' ','_')
    title = ''.join(e for e in title if e.isalnum())

    fname = str( str(datetime.today().strftime('%Y-%m-%d')) + '_' + str(title)+'.txt')
    with open(os.path.join('saved_chats', fname), 'w') as f:
        for message in history:
            f.write(message["role"]+ ': ' + message["content"]+'\n')
        f.close()


"""
MAIN SCRIPT
"""
### MAIN
if __name__=="__main__":
    print("### SETTING UP ENVIROMENT ###")
    print('loading whisper model...')
    whisper_model = whisper.load_model("medium") # pick the one that works best for you, but remember: only medium and large are multi language

    print('opening pygame ')
    pygame.mixer.init()
    
    ## IBM CLOUD
    ## 1. AUTH
    print('Authorizing IBM Cloud...')
    url ='your-ibm-cloud-service-url'
    apikey = 'your-ibm-cloud-api-key'
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

    # costants and knobs
    DEFAULT_CHAT = [{"role": "system", "content": "You are a helpful assistant and you will answer in paragraphs. A paragraph can be as long as 20 words."}]
    CHAT_LONG_ENOUGH = 4

    # vars init
    print("preparing a new conversation...\n")
    chat_history = DEFAULT_CHAT
    SleepTimer = -10000
    GoToSleep = True
    ConversationStarted = False

    while True:
        
        ## MIC TO TEXT
        GoToSleep = myaudio.record_to_file('output.wav',SleepTimer)

        # Many of these variables will be incorporated in an ASSISTANT class
        if GoToSleep: 
            SleepTimer = -10000
            if len(chat_history)>CHAT_LONG_ENOUGH: save_chat(chat_history)
            chat_history = DEFAULT_CHAT
            continue

        if not(ConversationStarted):
            SleepTimer = time.perf_counter()
            ConversationStarted= True

        try:
            question, detected_language = myaudio.whisper_wav_to_text('output.wav',whisper_model)
        except Exception as e:
            say(str(e))

        # check exit command
        if "THANKS" in question.upper():
            print('\nclosing chat...')
            save_chat(chat_history)
            SleepTimer = -10000
            ConversationStarted = False
        
        if "HEY" in question.upper() and "JARVIS" in question.upper() and detected_language=='en':
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
