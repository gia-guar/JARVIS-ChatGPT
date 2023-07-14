# import for prompt routing
from langchain import OpenAI
from langchain.agents import Tool
from langchain.agents import initialize_agent

from Assistant.research_mode import ResearchAssistant
from .Agents import generateReactAgent, generateGoogleAgent
import tiktoken

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
import torch

import Assistant.get_audio as myaudio
from .voice import *
from .tools import Translator, LocalSearchEngine, AssistantChat
from .tools import parse_conversation, count_tokens, take_last_k_interactions
from .webui import oobabooga_textgen

# imports for audio
import whisper
import wave
import pyaudio
import speech_recognition as sr
import time
import sys
from contextlib import contextmanager

#module used for speaking during recording
import webrtcvad
      
class VirtualAssistant:
    DEBUG = True
    DEFAULT_CHAT =  AssistantChat([{"role": "system", "content": "You are a helpful assistant. You can make question to make the conversation entertaining."}])
    RESPONSE_TIME = 1.5 #values that work well in my environment (ticks, not seconds)
    SLEEP_DELAY = 3 #seconds 
    MIN_RECORDING_TIME = .5 #seconds
    MAX_RECORDING_TIME = 60 #seconds
    VAD_AGGRESSIVENESS = 2  #1-3

    MAX_TOKENS = 4096

    DEVICE_INDEX = myaudio.detect_microphones()[0]
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = myaudio.get_device_channels()[DEVICE_INDEX]
    RATE = int(myaudio.get_devices()[DEVICE_INDEX]['defaultSampleRate'])

    print('using input device: ', myaudio.get_devices()[DEVICE_INDEX]['name'])

    CONVERSATION_LONG_ENOUGH = 4 #interactions (2 questions)

    def __init__(self, 
                 whisper_model=None, 
                 awake_with_keywords = ['elephant'],
                 model = "gpt-3.5-turbo",
                 embed_model = "text-embedding-ada-002",
                 translator_model = 'argostranslator',
                 
                 **kwargs):
        try:
            openai.api_key = kwargs['openai_api']
        except:
            print('OpenAI API key not found')
        
        # HEAVY STUFF FIRTS
        # Filling the GPU with the model
        if whisper_model == None:
            if 'whisper_size' not in kwargs: raise Exception('whisper model needs to be specified')
            self.interpreter = whisper.load_model(kwargs['whisper_size'])
        else:
            self.interpreter = whisper_model

        # STATUS and PROMPT ANALYZER
        if 'mode' in kwargs: 
            if kwargs['mode'].upper() != 'CHAT' and  kwargs['mode'].upper() != 'RESEARCH': raise KeyError()
            self.MODE = kwargs['mode']
        self.DIRECTORIES={
            'CHAT_DIR': os.path.realpath(os.path.join(os.getcwd(), 'saved_chats')),
            'SOUND_DIR':os.path.realpath(os.path.join(os.getcwd(), 'Assistant', 'sounds')),
            'VOICE_DIR':os.path.realpath(os.path.join(os.getcwd(), 'Assistant', 'voices'))
        }

        self.func_descript={
            "CHAT":[
                "tools: the prompt requires an action like handling a file, saving a conversation, changing some specified parameters...",
                "respond: provide an answer to a question",
                "you don't know the answer or you can't satisfy the request."],
            "RESEARCH":[
                "tools: the prompt requires one or multiple actions like reading a file, downloading a known resource",
                "respond: provide an answer based on scientific information",
            ]
        }


        # TEXT and VOICE
        if 'voice_id' in kwargs.keys():
            for item in kwargs['voice_id']:
                print(kwargs['voice_id'][item])
                kwargs['voice_id'][item] = (os.path.join(self.DIRECTORIES["VOICE_DIR"], kwargs['voice_id'][item])) + '.wav'
        else:
            kwargs['voice_id'] = os.path.join(self.DIRECTORIES["VOICE_DIR"], 'default.wav')

        self.languages = {
            'en': "English",
            'it': "Italian",
            # add yours
        }


        self.voice = Voice(write_dir = self.DIRECTORIES['SOUND_DIR'], languages = self.languages, **kwargs)
        self.translator = Translator(model=translator_model, translator_languages = list(self.languages.keys()))
        self.answer_engine = model

        self.search_engine = LocalSearchEngine(
            embed_model = embed_model, 
            tldr_model = kwargs['search_engine_llm'] if 'search_engine_llm' in list(kwargs.keys()) else model,
            translator_model=translator_model,
            translator_languages = list(self.languages.keys()))
        
        self.is_awake = False
        self.current_conversation = self.DEFAULT_CHAT

        # AUDIO 
        # initialize the VAD module
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(self.VAD_AGGRESSIVENESS) 
        
        if 'awake_with_keywords' in kwargs:
            self.Keywords = kwargs['awake_with_keywords']
        else:
            self.Keywords = awake_with_keywords
        self.Keywords = awake_with_keywords
        self.ears = sr.Recognizer()


        # init finished
        self.play('system_online_bleep.mp3')
    
    # STATUS ###############################################################################################
    def switch_mode(self):
        if self.MODE == 'CHAT': 
            self.say('Moving into research mode', VoiceIdx='en', elevenlabs=True)
            self.play('Sci-Fi-UI.mp3',loop=True)
            self.init_research_mode()
            pygame.mixer.stop()
            self.play('system_online_bleep.mp3', PlayAndWait=True)
            response =  self.translator.translate('research mode is ready', to_language=langid.classify(self.current_conversation[-1]['content']), from_language='en').lower()
            return response
        else:
            self.MODE = 'CHAT'
            response =  self.translator.translate('chat mode enabled', to_language=langid.classify(self.current_conversation[-1]['content'])[0],from_language='en').lower()
            return response

    def identify_explicit_command(self, prompt):
        prompt = self.translator.translate(prompt, to_language='en').lower()
        
        # if the prompt is long it's unlikely to be an explicit command 
        # (this condition prevents false positives)
        if len(prompt.split())>15: return
        
        INTERNET_COMMANDS = [
            "do an internet search",
            "look on the web",
            "do a web search",
            "control on the internet",
            "do a search",
            "make a search",
            "perform a search",
            "perform a web search"]
        
        if ("research mode" in prompt and self.MODE=='CHAT') or ("chat mode" in prompt and self.MODE=='RESEARCH'):
            print('found explicit command')
            return '-1'
        
        if self.MODE == 'CHAT':
            if any(command in prompt for command in INTERNET_COMMANDS):
                print('found explicit command')
                return '3'
        
        if self.MODE == 'RESEARCH':
            if "new workspace" in prompt or "new environment" in prompt:
                print('found explicit command')
                return '3'
        
    
    def use_tools(self, prompt, debug = DEBUG):
        if debug: print(' -use tools ')
        # tools for chat mode
        if self.MODE == "CAHT":
            ActionManager = generateReactAgent(self, k=1)
            return ActionManager.run(input = prompt)
        
        # research mode
        else:
            return self.ResearchAssistant.agent.run(input = prompt)
            
    
    def secondary_agent(self, prompt, debug = DEBUG): 
        if self.MODE == 'CHAT':
            if debug: print(' - web surfing ')   
            WebSurfingAgent = generateGoogleAgent(self, k=1)
            return WebSurfingAgent.run(prompt)
        if self.MODE == 'RESEARCH':
            if debug: print(' - assessing new workspace ') 
            return self.ResearchAssistant.PROTOCOL_begin_new_workspace(prompt)
        

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
        if len(self.current_conversation()) > self.CONVERSATION_LONG_ENOUGH:
            self.save_chat()

        self.play('sleep.mp3', PlayAndWait=True)

    # [stable]
    def analyze_prompt(self, prompt, debug = DEBUG):
        if debug: print(f' - analyzing prompt in {self.MODE} mode')
        # Hard coded options: DO this, Look on INTERNET...
        flag = self.identify_explicit_command(prompt)
        if flag is not None: return flag

        # CHAT MODE 
        if self.MODE == 'CHAT':
            context ="""You are a prompt manager. A number must always be present in your answer. You can perform some actions and decide which associated number is required. Your actions:"""
            for i, function in enumerate(self.func_descript['CHAT']):
                context += f"\n{i+1}) {function};"
            context += "\nYou can answer only with numbers. A number must always be present in your answer."
            context += """\nHere are some example:
            \nPROMPT: 'find and summarize all the files about history'\n1
            \nPROMPT: 'find a past conversation about planes'\n1
            \nPROMPT: 'do you agree?'\n2
            \nPROMPT: 'Salva questa conversazione'\n1
            \nPROMPT: 'How is the weather?'\n3
            \nPROMPT: 'credo sia giusto.'\n2
            \nPROMPT: '¿Cuál es la noticia de hoy?'\n3
            \nPROMPT: 'Thank you'\n2"""

            CHAT = [{"role": "system", "content": context},
                    {"role": "user", "content":f"PROMPT: '{prompt}'"}]

            flag = self.identify_explicit_command(prompt)

        # RESEARCH MODE    
        else:
            context ="""You are a prompt manager. A number must always be present in your answer. You can perform some actions and decide which associated number is required. You are designed to assist users with their academic research. You are equipped with a range of tools. Your tools:"""
            for i, function in enumerate(self.func_descript['RESEARCH']):
                context += f"\n{i+1}) {function};"
            context += "\nYou can answer only with numbers. A number must always be present in your answer."
            context += """\nHere are some example:
            \nPROMPT: 'begin a new project'\n1
            \nPROMPT: 'download papers about ...'\n1
            \nPROMPT: 'what are the mechanichal properties of carbon fiber?'\n2
            \nPROMPT: 'Salva questa conversazione'\n1
            \nPROMPT: 'What are the authors of the paper XYZ?'\n2
            \nPROMPT: 'What studies mention Transformers architectures?'\n2
            \nPROMPT: 'Find new papers that are similar to paper XYZ'\n1
            \nPROMPT: 'Tell me more'\n2
            \nPROMPT: 'what is up?\n2"""

            CHAT = [{"role": "system", "content": context},
                    {"role": "user", "content":f"PROMPT: '{prompt}'"}]


        if debug: print(' - - submitting request')
        response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    temperature=0,
                    max_tokens=10,
                    messages=CHAT)
        flag = response['choices'][0]['message']['content']
        if debug: print(' - - got answer')
        
        return flag

    # CONVERSATION ################################################################################

    def start_new_conversation(self):
        if len(self.current_conversation)>2: 
            print('forgetting the last conversation')
        self.current_conversation = self.DEFAULT_CHAT

    def expand_conversation(self, role, content): self.current_conversation.append({"role":role, "content":content})

    def get_answer(self, question, optimize_cuda = False, debug=DEBUG):
        if debug: print(' - thinking')
        if self.MODE == "CHAT":
            temp = copy.deepcopy(self.current_conversation())
            temp.append({"role":"user", "content":question})

            self.play('thinking.mp3', loop=True)

            if self.answer_engine == 'gpt-3.5-turbo':
                if debug: print(' - - submitting request')
                API_response = openai.ChatCompletion.create(
                    model=self.answer_engine,
                    messages=temp)
                answer = API_response['choices'][0]['message']['content']
                if debug: print(' - - got answer')
                
            
            elif self.answer_engine == 'anon8231489123_vicuna-13b-GPTQ-4bit-128g':
                lang_id = langid.classify(question)[0]
                if optimize_cuda:
                    # free space on the GPU
                    self.deallocate_whisper()
                # use GPU to process the answer
                answer = oobabooga_textgen(prompt = temp)
                answer = self.translator.translate(answer, from_language=langid.classify(answer)[0], to_language=lang_id)
                if optimize_cuda:
                    # try to get the model back to GPU
                    self.allocate_whisper()

            elif self.answer_engine == 'eachadea_ggml-vicuna-13b-4bit':
                answer = oobabooga_textgen(prompt = question)
        
        # RESEARCH MODE
        else:
            if self.ResearchAssistant.query_engine == None:
                return 'error: no workspace loaded. I cannot provide precise information without a workspace loaded on research mode'
            res = self.ResearchAssistant.query_engine.query(question)
            answer = res.response
        pygame.mixer.stop()

        self.expand_conversation(role="assistant", content=answer)

        self.last_interaction = time.perf_counter()
        if debug: print(' - - finished')
        return answer

    def save_chat(self, debug = DEBUG):
        if debug: print(' - saving chat')
        if not os.path.isdir(self.DIRECTORIES['CHAT_DIR']): os.mkdir(self.DIRECTORIES['CHAT_DIR'])

        if not self.current_conversation.is_saved(): 
            if debug: print(' - - generating title')
            title = self.get_answer(question="generate a very short title for this conversation")
            self.say(f'I am saving this conversation with title: {title}', VoiceIdx='en', IBM=False, elevenlabs=True)

            self.play('data_writing.mp3', PlayAndWait=True)
            
            prompt =  [{"role": "system", "content": "You don't like redundancy and use as few words as possible"},
                {"role":"user", "content":f"Associate a tag to this title: {title} \nHere is an example: 'Exploring Text to Speech Popular Techniques and Deep Learning Approaches' is associated to 'Deep Learning'"}]
            
            if debug: print(' - - submitting request')
            API_response = openai.ChatCompletion.create(
                        model='gpt-3.5-turbo',
                        max_tokens=5,
                        temperature=0,
                        messages=prompt)
            if debug: print(' - - got answer')
            if debug: print(' - - processing response')
            answer = API_response['choices'][0]['message']['content']
            answer = re.sub(r'[^\w\s]', '',answer)
            answer = re.sub(' ', '',answer)
            fname = str( str(datetime.today().strftime('%Y-%m-%d')) + '_' + str(answer)+'.txt')
            self.current_conversation.filename = fname
        else:
            self.say(f'I am overwriting the conversation {fname}', VoiceIdx='en', IBM=False, elevenlabs=True)
            fname = self.current_conversation.filename

        with open(os.path.join(self.DIRECTORIES['CHAT_DIR'], fname), 'w') as f:
            for message in self.current_conversation():
                f.write(message["role"]+ ': ' + message["content"]+'\n')
            f.close()
        
        self.is_awake = False
        return f"file: {os.path.join(self.DIRECTORIES['CHAT_DIR'], fname)} saved successfully"


    # ACTIONS ##################################################################################
    def init_research_mode(self, workspace=None):
        
        if workspace is None:
            # get last created workspace
            if 'workspaces' in os.listdir(os.getcwd()):
                search_dir = os.path.join('workspaces')
                subdirs = os.listdir(search_dir)
                subdirs.sort(key=lambda fn: os.path.getmtime(os.path.join(search_dir, fn)))
                subdirs.reverse()
                for subd in subdirs:
                    folder_path = os.path.join('workspaces',subd)
                    if os.path.isdir(folder_path):
                        self.say('loading the last created workspace', VoiceIdx='en', elevenlabs=True)
                        workspace = os.path.abspath( folder_path )
                        break

        self.play('Sci-Fi-UI.mp3',loop=True)
        self.MODE = 'RESEARCH'
        self.ResearchAssistant = ResearchAssistant(
            current_conversation=self.current_conversation,
            index_name='paperquestioning',
            workspace=workspace)
        pygame.mixer.stop()
        
    def deallocate_whisper(self):
        model_name = self.interpreter.name
        model_current_device = self.interpreter.device

        self.interpreter = None
        torch.cuda.empty_cache()

        if model_current_device.type == 'cuda':
            print('loading Whisper model to cpu')
            self.interpreter = whisper.load_model(model_name, device='cpu')
        torch.cuda.empty_cache()

    def allocate_whisper(self):
        model_name = self.interpreter.name
        model_current_device = self.interpreter.device
        self.interpreter = None

        torch.cuda.empty_cache()
        if model_current_device.type == 'cpu':
            try:
                torch.cuda.empty_cache()
                print('loading Whisper model to CUDA')
                self.interpreter = whisper.load_model(model_name, device='cuda')
            except:
                self.interpreter = None
                print(f"cuda dedicated memory isufficient: {torch.cuda.memory_allocated()/1e6} GB already occupuied")
                print(f"keeping Whisper model to cpu")
                self.interpreter = whisper.load_model(model_name, device='cpu')
        torch.cuda.empty_cache()
                
    def switch_whisper_device(self):
        model_name = self.interpreter.name
        model_current_device = self.interpreter.device

        self.interpreter = None
        torch.cuda.empty_cache()

        if model_current_device.type == 'cuda':
            print('loading Whisper model to cpu')
            self.interpreter = whisper.load_model(model_name, device='cpu')
            torch.cuda.empty_cache()
        else:
            try:
                torch.cuda.empty_cache()
                print('loading Whisper model to CUDA')
                self.interpreter = whisper.load_model(model_name, device='cuda')
            except:
                print(f"cuda dedicated memory isufficient: {torch.cuda.memory_allocated()/1e6} GB already occupuied")
                print(f"keeping Whisper model to cpu")
                self.interpreter = whisper.load_model(model_name, device='cpu')
                torch.cuda.empty_cache()
    
    def open_file(self, filename, debug=DEBUG):
        if debug: print(' - opening file')
        # look for the file
        file = None
        for fname in os.listdir(self.DIRECTORIES['CHAT_DIR']):
            
            # look for sub-strings (in case extension is forgotten)
            if filename in fname:
                file = open(os.path.join(self.DIRECTORIES['CHAT_DIR'], filename), 'r')
                file = file.read()

        if file is None: return 'No such file'

        return file  
   

    def find_file(self, keywords, n=3, debug=DEBUG):
        if debug: print(' -finding file')
        #self.play('thinking.mp3', loop=True)
        summary = self.search_engine.accurate_search(key=keywords, from_csv=True, n=n)
        # self.play('wake.mp3')

        response = ''
        for i in range(n):
            response += f"\nFilename: {summary.file_names[i]} ; Topics discussed: {summary.tags[i]}" 
        return response




    # SPEAK ####################################################################################
    def play(self, fname, PlayAndWait=False, loop=False, debug = DEBUG):
        if loop: loop=-1
        else: loop = 0

        if pygame.mixer.get_init() is None: pygame.mixer.init()
        if debug: print(' - playing')
        try:
            pygame.mixer.music.load(os.path.join(self.DIRECTORIES["SOUND_DIR"], fname))
        except Exception as e:
            print(e)
            return
        pygame.mixer.music.set_volume(0.5)
        pygame.mixer.music.play(loops=loop)

        if PlayAndWait:
            while(pygame.mixer.music.get_busy()):pass
        if debug: print(' - -  finihed playing')

    def say(self, text, VoiceIdx='jarvis', elevenlabs=False, IBM=False):  
        langIdx = langid.classify(text)[0]

        print(f"[Assistant]: {text}")
        if elevenlabs and IBM: raise(Exception('IBM and ElevenLabs can t be both true'))
        
        if elevenlabs:
            try:
                try: 
                    self.voice.speak(text=text, VoiceIdx=langIdx, elevenlabs=True, IBM=False, mode='online')
                    return
                except Exception as e:
                    print(f"couldn t speak with: {e}")
                    self.voice.speak(text=text, VoiceIdx=langIdx, elevenlabs=False, IBM=True, mode='online')
                    return
            except:
                self.voice.speak(text=text, VoiceIdx=VoiceIdx, elevenlabs=False, IBM=False, mode='offline')
                return
        
        elif IBM:
            try:
                try: 
                    self.voice.speak(text=text, VoiceIdx=langIdx, elevenlabs=False, IBM=True, mode='online')
                    return
                except:
                    self.voice.speak(text=text, VoiceIdx=langIdx, elevenlabs=True, IBM=False, mode='online')
                    return
            except:
                self.voice.speak(text=text, VoiceIdx=VoiceIdx, elevenlabs=False, IBM=False, mode='offline')
                return
        
        try:
            self.voice.speak(text=text, VoiceIdx='jarvis',elevenlabs=False, IBM=False, mode='offline')
        except Exception as e:
            self.voice.speak(text=text, VoiceIdx=langIdx, elevenlabs=False, IBM=False, mode='offline')
            print(VoiceIdx, elevenlabs, IBM)
            print(e)
            raise Exception('No such specifications')


    # LISTEN #############################################################################################

    
    #function that blocks the code until the wakeword, or wakewords are encountered
    def block_until_wakeword(self, verbosity=False):        
        if verbosity: print("listening passively...", end="")

        from struct import unpack_from
        import pvporcupine

        #initialize values
        porcupine = None
        pa = None
        audio_stream = None

        try:
            porcupine = pvporcupine.create(access_key=os.environ["PORCUPINE_KEY"], 
                                        keywords=self.Keywords)

            pa = pyaudio.PyAudio()

            audio_stream = pa.open(
                rate=porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=porcupine.frame_length)

            #not strictly necessary, but helps debug if something overwrote the keywords
            print(f"Listening for wake word '{self.Keywords[0]}'...")

            #loop to preform while waiting(does not noticeably use the CPU)
            while True:
                pcm = audio_stream.read(porcupine.frame_length)
                pcm = unpack_from("h" * porcupine.frame_length, pcm)

                keyword_index = porcupine.process(pcm)

                #same actions activated as previous function
                #NOTE: keyword_index is -1 unless wakeword encountered, then it's the index of the wakeword in the list
                #(different wakewords activate different profiles?)
                if keyword_index >= 0:
                    print("wakeword encountered")
                    self.start_new_conversation()
                    self.play('wake.mp3',PlayAndWait=False)
                    self.is_awake = True
                    return
        finally:
            #clean up
            if audio_stream is not None:
                audio_stream.close()
            if pa is not None:
                pa.terminate()
            if porcupine is not None:
                porcupine.delete()
     
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
        wf = wave.open(file_path, 'wb', )
        wf.setnchannels(self.CHANNELS)
        sample_width = pyaudio.PyAudio().get_sample_size(self.FORMAT)
        wf.setsampwidth(sample_width)
        frames = self.record()
        
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
    
    def record(self):
        # Your current setup
        vad_rate = 32000
        frame_length_ms = 20
        vad_CHUNK = (vad_rate * frame_length_ms) // 1000


        p = pyaudio.PyAudio()
        vad_stream = p.open(format=self.FORMAT,
                        channels=1,
                        rate=vad_rate,
                        input=True,
                        frames_per_buffer=vad_CHUNK)
        
        rec_stream = p.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        frames_per_buffer=self.CHUNK)

        frames = []
        try:
            silence_time = 0
            speaked = False
            is_voice = False
            print("listening...")
            
            start_time = time.perf_counter()
            while True:
                rec_data = rec_stream.read(self.CHUNK)
                frames.append(rec_data)

                # detect voice activity
                data = vad_stream.read(vad_CHUNK) 
                                       
                try:
                    is_voice = self.vad.is_speech(data, vad_rate) 
                except Exception as e:
                    print(f"Error during VAD: {e}")

                # Calculate time since the last voice activity
                if is_voice and (time.perf_counter()-start_time)>self.MIN_RECORDING_TIME:
                    speaked = True
                    silence_time = 0
                else:
                    silence_time += frame_length_ms / 1000

                # Print debugging information (useful for tuning sensitivity)

                # Stop recording if silence duration exceeds the threshold or if the time limit is reached
                if (silence_time > self.RESPONSE_TIME and speaked) or (time.perf_counter() - start_time > self.MAX_RECORDING_TIME):
                    break

                if silence_time > self.MAX_RECORDING_TIME:
                    self.go_to_sleep()
                    break
                
                time.sleep(frame_length_ms / 10000)

        except KeyboardInterrupt:
            print("Done recording")
        except Exception as e:
            print(str(e))
            print(silence_time,self.RESPONSE_TIME,self.SLEEP_DELAY)
            exit()

        vad_stream.stop_stream()
        vad_stream.close()
        rec_stream.stop_stream()
        rec_stream.close()
        p.terminate()
        return frames
    

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

