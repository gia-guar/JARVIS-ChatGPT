# imports for chats
import  pygame
import os

from datetime import datetime
import copy
import openai
import time

from Assistant.voice import *
from Assistant.tools import Translator, LocalSearchEngine

# imports for audio
import whisper
import wave
import pyaudio
import speech_recognition as sr
import audioop
import math
import time
      
class VirtualAssistant:
    DEFAULT_CHAT =  [{"role": "system", "content": "You are a helpful assistant and you will answer in paragraphs. A paragraph can be as long as 20 words."}]
    RESPONSE_TIME = 3 #seconds
    SLEEP_DELAY = 30 #seconds

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    SILENCE_THRESHOLD = 1500

    CONVERSATION_LONG_ENOUGH = 4 #interactions (2 questions)


    def __init__(self, 
                 whisper_model, 
                 awake_with_keywords = ['elephant'],
                 model = "gpt-3.5-turbo",
                 embed_model = "text-embedding-ada-002",
                 **kwargs):
        try:
            openai.api_key = kwargs['openai_api']
        except:
            print('OpenAI API key not found')

        self.DIRECTORIES={
            'CHAT_DIR': os.path.realpath(os.path.join(os.getcwd(), 'saved_chats')),
            'SOUND_DIR':os.path.realpath(os.path.join(os.getcwd(), 'Assistant', 'sounds'))
        }

        # TEXT 
        self.voice = Voice(**kwargs)
        self.translator = Translator(model=model)
        self.answer_engine = model
        self.search_engine = LocalSearchEngine(
            embed_model = embed_model, 
            tldr_model = model)
        
        self.languages = {
            'en': "English",
            'it': "Italian",
            # add your
        }

        self.is_awake = False
        self.current_conversation = self.DEFAULT_CHAT

        # AUDIO 
        if 'awake_with_keyowords' in kwargs:
            self.Keywords = kwargs['awake_with_keywords']
        else:
            self.Keywords = awake_with_keywords

        self.ears = sr.Recognizer()
        self.interpreter = whisper_model

        
        # init finished
        self.play('system_online_bleep.mp3')
    
    # STATUS ###############################################################################################

    def set_params(**kwargs):
        for key in kwargs:   
            exec("self.%s = %d" % (key,kwargs[key]))

    def set_directories(self, **kwargs):
        for item in kwargs:
            try:
                if not(os.path.isdir(kwargs[item])): raise Exception

                print(f'updating {item} from {self.DIRECTORIES[item.upper()]} == to => {kwargs[item]}')
                self.DIRECTORIES[item.upper()] = kwargs[item]
                
            except:
                self.play('error.mp3', PlayAndWait=True)
                print(f"{kwargs[item]}: not found")


    def go_to_sleep(self):
        print('[Assistant going to sleep]')
        self.is_awake = False
        if len(self.current_conversation) > self.CONVERSATION_LONG_ENOUGH:
            self.save_chat()

        self.play('sleep.mp3', PlayAndWait=True)


    # CONVERSATION ################################################################################

    def start_new_conversation(self):
        if len(self.current_conversation)>2: 
            print('forgetting the last conversation')
        self.current_conversation = self.DEFAULT_CHAT

    def get_answer(self, question, update=True):
        if update==True:
            self.current_conversation.append({"role":"user", "content":question})
            temp = self.current_conversation
        else:
            temp = copy.deepcopy(self.current_conversation)
            temp.append({"role":"user", "content":question})

        self.play('thinking.mp3')
        API_response = openai.ChatCompletion.create(
            model=self.answer_engine,
            messages=temp)
        
        answer = API_response['choices'][0]['message']['content']
        self.current_conversation.append({"role": "assistant", "content":answer})

        self.last_interaction = time.perf_counter()
        pygame.mixer.music.stop()

        return answer


    def save_chat(self):
        if not os.path.isdir(self.DIRECTORIES['CHAT_DIR']): os.mkdir(self.DIRECTORIES['CHAT_DIR'])
        
        title = self.get_answer(question="generate a title for this conversation", update=False)
        self.say(f'I am saving this conversation with title: {title}', VoiceIdx='en')

        self.play('data_writing.mp3', PlayAndWait=True)

        title = title.replace(' ','_')
        title = ''.join(e for e in title if e.isalnum())

        fname = str( str(datetime.today().strftime('%Y-%m-%d')) + '_' + str(title)+'.txt')
        with open(os.path.join(self.DIRECTORIES['CHAT_DIR'], fname), 'w') as f:
            for message in self.current_conversation:
                f.write(message["role"]+ ': ' + message["content"]+'\n')
            f.close()

    # SPEAK ####################################################################################
    def play(self, fname, PlayAndWait=False):
        if pygame.mixer.get_init() is None: pygame.mixer.init()

        try:
            pygame.mixer.music.load(os.path.join(self.DIRECTORIES["SOUND_DIR"], fname))
        except Exception as e:
            print(e)
            return
        pygame.mixer.music.set_volume(0.5)
        pygame.mixer.music.play()

        if PlayAndWait:
            while(pygame.mixer.music.get_busy()):pass

    def say(self, text, VoiceIdx='jarvis'):    
        ## Pick the right voice
        if VoiceIdx != 'jarvis':
            # Try online 
            try:    
                if not os.path.isdir('saved_chats'): os.mkdir("saved_chats")
                with open(os.path.join(os.getcwd(),'Assistant' 'answers','speech.mp3'),'wb') as audio_file:
                    res = self.voice.tts_service.synthesize(text, accept='audio/mp3', voice=get_ibm_voice_id(VoiceIdx)).get_result()
                    audio_file.write(res.content)

                ## SAY OUTLOUD
                if pygame.mixer.get_init() is None:
                    pygame.mixer.init()
                pygame.mixer.music.load("./answers/speech.mp3")
                pygame.mixer.music.set_volume(0.5)
                pygame.mixer.music.play()
                while(pygame.mixer.music.get_busy()): pass
                pygame.mixer.music.load(os.path.join('voices','empty.mp3'))
                os.remove("./answers/speech.mp3")
            
            # go offline if fail
            except:
                print('*!* IBM credit likely ended *!*  > using pyttsx3 for voice generation)')
                print(f'\n[assistant]: {text}')
                
                self.voice.offline = self.change_offline_lang(lang_id=VoiceIdx)
                self.voice.offline.say(text)
                self.voice.offline.runAndWait()

        else:
            self.voice.synthetic_voice.synthesize(text)
            self.voice.synthetic_voice.vocode()


    # works with pyttsx3 text to speech engine
    def change_offline_lang(self, lang_id):
        
        for voice in self.voice.offline.getProperty('voices'):
            if self.languages[lang_id] in voice.name:
                self.voice.offline.setProperty('voice', voice.id)
                return self.voice.offline
            
        self.say('This language does not belong to my principles, please add it', VoiceIdx='en')
    

    # LISTEN #############################################################################################
    def listen_passively(self, verbosity=False):
        with sr.Microphone() as source:
            if verbosity: print("listenting passively...", end="")
            audio = self.ears.listen(source)
            query = ''

            try: 
                query = self.ears.recognize(audio)
                if verbosity: print(f"user said: {query}")
            except Exception as e:
                if verbosity: print(str(e))
        
        # if any keyword is present in the query return True (awake the assistant)
        if any(keyword in query.split() for keyword in self.Keywords):
            self.start_new_conversation()
            self.play('wake.mp3',PlayAndWait=False)
            self.is_awake = True
        
    def record_to_file(self, file_path):
        wf = wave.open(file_path, 'wb')
        wf.setnchannels(self.CHANNELS)
        sample_width, frames = self.record()
        wf.setsampwidth(sample_width)
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
    
    def record(self):
        ## init Microphone streamline
        p = pyaudio.PyAudio()
        stream = p.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        frames_per_buffer=self.CHUNK)
        
        frames = []
        try:
            silence_time = 0
            speaked = False
            print("listening...")

            while True:
                delta = time.perf_counter()
                data = stream.read(self.CHUNK)
                frames.append(data)
                    
                # detect silence
                data = stream.read(self.CHUNK)
                sound_amplitude = audioop.rms(data, 2)
                delta = time.perf_counter() - delta

                if(sound_amplitude < self.SILENCE_THRESHOLD):      
                    silence_time = silence_time + delta

                    # break the loop and return the audio
                    if (silence_time > self.RESPONSE_TIME) and speaked:
                        raise KeyboardInterrupt   
                    
                    if silence_time > self.SLEEP_DELAY:
                        self.go_to_sleep()
                        raise KeyboardInterrupt 
                else:
                    speaked = True
                    silence_time = 0

        except KeyboardInterrupt:
            print("Done recording")
        except Exception as e:
            print(str(e))
            print(silence_time,self.RESPONSE_TIME,self.SLEEP_DELAY)

        sample_width = p.get_sample_size(self.FORMAT)
        stream.stop_stream()
        stream.close()
        p.terminate()
        return sample_width, frames