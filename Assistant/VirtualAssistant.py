# imports for chats
import  pygame
import os
import re

import pandas as pd
from datetime import datetime
import copy
import openai
import time
import langid
import Assistant.get_audio as myaudio

from .voice import *
from .tools import Translator, LocalSearchEngine

# imports for audio
import wave
import pyaudio
import speech_recognition as sr
import audioop
import time
import sys
from contextlib import contextmanager 
      
class VirtualAssistant:
    DEFAULT_CHAT =  [{"role": "system", "content": "You are a helpful assistant. You can make question to make the conversation entertaining."}]
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

        # STATUS
        self.DIRECTORIES={
            'CHAT_DIR': os.path.realpath(os.path.join(os.getcwd(), 'saved_chats')),
            'SOUND_DIR':os.path.realpath(os.path.join(os.getcwd(), 'Assistant', 'sounds')),
            'VOICE_DIR':os.path.realpath(os.path.join(os.getcwd(), 'Assistant', 'voices'))
        }

        self.functions=[
            'find a file: file can store past conversations, textual information',
            'respond: provide an answer to a question',
            'adjust settings',
            'find a directory inside the Computer',
            'save this conversation for the future',
            'start a new conversation and delete the current one',
            'check on internet',
            'None of the above',
        ]
        

        # TEXT and VOICE
        if 'voice_id' in kwargs.keys():
            kwargs['voice_id'] = os.path.join(self.DIRECTORIES["VOICE_DIR"], kwargs['voice_id']+'.wav')
        else:
            kwargs['voice_id'] = os.path.join(self.DIRECTORIES["VOICE_DIR"], 'default.wav')

        self.voice = Voice(**kwargs)
        self.translator = Translator(model=model)
        self.answer_engine = model
        self.search_engine = LocalSearchEngine(
            embed_model = embed_model, 
            tldr_model = model,
            translator_model='translators')
        
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

    def analyze_prompt(self):
        context ="""You are a prompt manager. A number must always be present in your answer. You have access to a suite of actions and decide which associated number is required. Your actions:"""
        for i, function in enumerate(self.functions):
            context += f"\n{i+1} - {function};"
        context += "\nYou can answer only with numbers. A number must always be present in your answer."

        CHAT = [{"role": "system", "content": context},
                {"role": "user", "content":f"PROMPT: '{self.current_conversation[-1]['content']}'"}]

        response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    temperature=0,
                    max_tokens=10,
                    messages=CHAT)
        
        return response['choices'][0]['message']['content']




    # CONVERSATION ################################################################################

    def start_new_conversation(self):
        if len(self.current_conversation)>2: 
            print('forgetting the last conversation')
        self.current_conversation = self.DEFAULT_CHAT

    def expand_conversation(self, role, content): self.current_conversation.append({"role":role, "content":content})

    def get_answer(self, question, update=False):
        if update==True:
            self.expand_conversation(role="user", content=question)
            temp = self.current_conversation
        else:
            temp = copy.deepcopy(self.current_conversation)
            temp.append({"role":"user", "content":question})

        self.play('thinking.mp3')
        API_response = openai.ChatCompletion.create(
            model=self.answer_engine,
            messages=temp)
        
        answer = API_response['choices'][0]['message']['content']
        self.expand_conversation(role="assistant", content=answer)

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




    # ACTIONS ##################################################################################

    def confirm_choice(self, confirm_question, lang_id=None):
        prompt = self.current_conversation[-1]["content"]
        if lang_id is None: lang_id = langid.classify(prompt)[0]
        confirm_question = self.translator.translate(confirm_question, lang_id)       

        self.say(confirm_question, VoiceIdx=lang_id)
        self.record_to_file('output.wav')
        response, _ = myaudio.whisper_wav_to_text('output.wav',self.interpreter)
        
        if self.search_engine.compute_similarity(key='yes',text=response)>0.8:
            self.expand_conversation(role="assistant", content=confirm_question)
            self.expand_conversation(role="user", content=response)
            return True
        else:
            return False


    def find_file(self):
        prompt = self.current_conversation[-1]["content"]
        lang_id = langid.classify(prompt)[0]

        # confirm choice first:
        confirm_question = "I am about to begin a search, should I proceed?"
        if not(self.confirm_choice(confirm_question, lang_id = lang_id)):
            self.play('aborting.mp3')
            return -1
        
        # provide keywords:
        provide_tag_question = "Provide the argument of the search"
        provide_tag_question = self.translator.translate(provide_tag_question, lang_id)
        self.say(provide_tag_question, VoiceIdx=lang_id)
        self.record_to_file('output.wav')
        response, _ = myaudio.whisper_wav_to_text('output.wav',self.interpreter)
        self.expand_conversation(role="assistant", content=provide_tag_question)
        self.expand_conversation(role="user", content=response)

        if len(response)<5:
            # return to main if silence
            return
        
        keywords = re.sub('[^0-9a-zA-Z]+', ' ', response)
        keywords = keywords.split()

        self.play('thinking.mp3', loop=True)
        summary = self.search_engine.accurate_search(key=keywords, from_csv=True)
        self.play('wake.mp3')

        lang_id = langid.classify(prompt)[0]
        text = self.translator.translate("Research completed:", lang_id)
        
        all_files = pd.read_csv(os.path.join(self.search_engine.default_dir,'DATAFRAME.csv'))


        for i in range(len(summary.file_names)):
            try:
                idx = all_files.file_names.index(summary.file_names[i])
                topics = all_files.tags[idx]
            except:
                topics = f'error - {summary.file_names[i]} not found in {all_files.file_names}'

            text += f"\n - {i+1}): {topics.split()[:3]};"

        self.say(self.translator.translate(f'In the most relevant conversation the following topic were discussed: {summary.tags[0]}', lang_id),
            VoiceIdx=lang_id)

        print("\n")

        self.expand_conversation(role="assistant", content=text)

        # work in progress: asking to actually load the conversation 
        # work in progress: making recovering the last conversation more straightfoward
        return summary



        


    # SPEAK ####################################################################################
    def play(self, fname, PlayAndWait=False, loop=False):
        if loop: loop=-1
        else: loop = 0

        if pygame.mixer.get_init() is None: pygame.mixer.init()

        try:
            pygame.mixer.music.load(os.path.join(self.DIRECTORIES["SOUND_DIR"], fname))
        except Exception as e:
            print(e)
            return
        pygame.mixer.music.set_volume(0.5)
        pygame.mixer.music.play(loops=loop)

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
                print('*!* IBM credit likely ended *!*  > using pyttsx3 for voice generation')
                print(f'\n[assistant]: {text}')
                
                try:
                    self.voice.offline = self.change_offline_lang(lang_id=VoiceIdx)
                    self.voice.offline.say(text)
                    self.voice.offline.runAndWait()
                except:
                    text = self.translator.translate(input=text, language='english')
                    self.voice.offline = self.change_offline_lang(lang_id='en')
                    self.voice.offline.say(text)
                    self.voice.offline.runAndWait()

        else:
            print(f'\n[assistant]: {text}')   
            with suppress_stdout():
                self.voice.synthetic_voice.tts_to_file(text=text, speaker_wav=self.voice.path, language="en", file_path=os.path.join(self.DIRECTORIES['SOUND_DIR'], 'last_answer.wav'))
                self.play(os.path.join(self.DIRECTORIES['SOUND_DIR'], 'last_answer.wav'), PlayAndWait=True)

    # allows pyttsx3's text to speech engine to change language
    def change_offline_lang(self, lang_id):
        try:
            for voice in self.voice.offline.getProperty('voices'):
                if self.languages[lang_id] in voice.name:
                    self.voice.offline.setProperty('voice', voice.id)
                    return self.voice.offline
        except Exception as e:      
            print('error: ',e)
    

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
    

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout