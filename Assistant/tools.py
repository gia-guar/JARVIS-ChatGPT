# imports for Local Search Engine
import openai
import os
import pandas as pd
import numpy as np
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity
from tqdm import tqdm
import ast

from . import webui

# import for Translator
import regex as re
import langid
from textblob import TextBlob
try: import translators as ts
except: print('could not import translators package')
import argostranslate.package
import argostranslate.translate

import math
import time 
import collections

"""
AssistantChat: dictionary on steroids.
"""
class AssistantChat(collections.MutableSequence):
    def __init__(self, begin:list, *args):
        self.body = begin
        self.filename= None
        self.extend(list(args))
    
    def is_saved(self):
        return True if self.filename != None else False
    
    def insert(self, i, v):
        self.body.insert(i, v)

    def append(self, item):
        self.body.append(item)

    def __call__(self):
        return self.body

    def __len__(self): return len(self.body)

    def __getitem__(self, i): return self.body[i]

    def __delitem__(self, i): del self.body[i]

    def __setitem__(self, i, v):
        self.body[i] = v

    def __str__(self):
        return str(self.body)


"""
Translator: 
performs basic translation opration using ChatGPT. 
Setting temperature to 0 allows better raw results
"""

"""
options:
 - gpt-3.5-turbo: reasonably fast, online, requires openai credit usage 
 - translators 5.6.3 lib: online, excellent, long lags might occcur
 - [default] argostranslator: fast,  offline 
"""

class Translator:
    def __init__(self, model="argostranslator", **kwargs):
        POSSIBLE_MODELS = ['argostranslator','gpt-3.5-turbo', 'translators']
        if model not in POSSIBLE_MODELS:
            raise Exception('this Translation model is not available')
        
        self.DEFAULT_CHAT = [{"role": "system", 
                    "content": "You are a translator. You recieve text and target language as inputs and translate the text to the target language"}]
        self.body = None
        self.model = model
        langs = kwargs['translator_languages']
        self.languages = langs

        # Download and install Argos Translate packages
        argostranslate.package.update_package_index()
        available_packages = argostranslate.package.get_available_packages()
        
        langid.set_languages(langs)

        for i in range(len(langs)):
            for j in range(len(langs)):
                if langs[i]==langs[j]: continue
                
                try:
                    package_to_install = next(
                        filter(
                            lambda x: x.from_code == langs[i] and x.to_code == langs[j], available_packages
                        )
                    )
                except:
                    print(f'failed to add {langs[i]} => {langs[j]}')
                print(f'downloading Argos Translate Language packages...')
                try:
                    argostranslate.package.install_from_path(package_to_install.download())
                except:
                    pass
        

    def translate(self, input, to_language, from_language=None):
        if from_language == to_language: return input

        if from_language == None:
            from_language = langid.classify(input)[0]
            
        if self.model=="gpt-3.5-turbo":
            self.body = self.DEFAULT_CHAT
            self.body.append({"role":"user", "content":f"translate in {to_language}:'{input}'"})
            try:
                API_response = openai.ChatCompletion.create(
                        model=self.model,
                        temperature=0,  
                        messages=self.body)
                
            except Exception as e:
                print(f"couldn't translate {self.body[-1]}")
                print(e)
                return input
            return API_response['choices'][0]['message']['content']
        
        if self.model=='translators':
            try:
                res = ts.translate_text(input, translator='google', to_language=to_language, from_language=from_language)
            except:
                res = input
                self.model = 'argostranslator'
                print('translation using translators switching to argostranslate')
                
            return res
        
        if self.model == 'argostranslator':
            try:
                res = argostranslate.translate.translate(input, from_code=from_language, to_code=to_language)
            except:
                print(f"translation using argostranslate from: {from_language} - to -> {to_language} Failed")
                print(input)
                res= input
            return res

    

"""
LocalSearchEngine:
 - Looks for files in a foder;
 - extracts information;
 - create high value contents that allow for accurate search;

to be implemented:
 - extend reserarch to .pdf and .jpeg (w/ ChatGPT4)
    - extends also to videos;
    - extends also to scientific papers;
"""

class LocalSearchEngine:
    def __init__(self, 
                 embed_model = "text-embedding-ada-002", 
                 tldr_model = "gpt-3.5-turbo",
                 translator_model = "argostranslator",
                 translator_languages = ['en','it','es'],
                 default_dir = os.path.realpath(os.path.join(os.getcwd(),'saved_chats')),
                 irrelevancy_th=0.8):
        
        self.translate_engine = Translator(model=translator_model, translator_languages=translator_languages)
        self.tldr_model  = tldr_model
        self.embed_model = embed_model
        self.default_dir = default_dir
        self.irrelevancy_threshold = irrelevancy_th
    
    def compute_similarity(self, key, text):
        if type(key)==str:  key_embedding = self.compute_embeds(key)
        else: key_embedding = key
        
        if type(text)==str: query_embedding =self.compute_embeds(text)
        else: query_embedding = text

        similarity = cosine_similarity(key_embedding, query_embedding)
        return similarity
    

    def accurate_search(self, key, path=None, n=-1, from_csv=False):
        if path is None:
            path = self.default_dir

        print('\n')
        if 'DATAFRAME.csv' not in os.listdir(path):
            print('> > DATAFRAME.csv not detected building a new one')
            pd.DataFrame({'file_names':['DATAFRAME.csv'], 'similarity':[0],"tags":[None]}).to_csv(os.path.join(path, 'DATAFRAME.csv'))

        if isinstance(key, list) or isinstance(key, tuple):
            key = " ".join(key)
        
        # USE EXISTING DATAFRAME TO MAKE SEARCH FASTER (skip tag generation)
        if from_csv:
            DataFrame = pd.read_csv(os.path.join(path,'DATAFRAME.csv'))
            fnames = DataFrame["file_names"]
            tags = DataFrame["tags"]
            embeds = DataFrame["embeddings"]

            if len(fnames)!=len(os.listdir(path)):
                print('> dataset not updated. Updating it now...')
                
                self.produce_folder_tags() ### I should add a parameter to specify HugginFaceHub (free) embeddings or OpenAI ones ($)
        
        print('> Analyzing DataFrame:')

        results = []
        topics = []
        
        key_embed = {}
        for lang in self.translate_engine.languages:
            transl_key = self.translate_engine.translate(input=key, to_language=langid.classify(lang)[0], from_language=langid.classify(key)[0])
            print(f'> > computing key embedding in {lang} language')
            key_embed[lang]= self.compute_embeds(transl_key)

        for i in tqdm(range(len(fnames))):
            if not(fnames[i].endswith('.txt')):
                results.append(0)
                topics.append('None')
                continue
            
            # extract tags associated to the file
            file_tags = tags[i]
            topics.append(file_tags)

            # extract and parse the saved embeddings
            file_embeds = ast.literal_eval( embeds[i] ) # from "[a, b, c,]" to [a, b, c]

            # take the key embedding from the same language (more accurate) 
            key_embedding = key_embed[langid.classify(file_tags)[0]]
            
            done=False
            while not(done):
                try:
                    relevance = self.compute_similarity(file_embeds, key_embedding)
                    done=True

                except Exception as e:
                    print(e)

            results.append(relevance)

        if n==-1: n=len(fnames)
        df = pd.DataFrame({'file_names':fnames, 'similarity':results,"tags":topics})
        df = df.sort_values(by='similarity', ascending=False)
        df = df.reset_index(drop=True)

        return df.head(n)


    def produce_folder_tags(self, path=None):
        if path is None:
            path = self.default_dir
        
        if ('DATAFRAME.csv' in os.listdir(path)):
            print('> > DataFrame existing')
        else:
            print('> > Creating empty DataFrame')
            pd.DataFrame(columns=['file_names', 'tags', 'embeddings']).to_csv(os.path.join(path,'DATAFRAME.csv'))

        existing_df = pd.read_csv(os.path.join(path, 'DATAFRAME.csv'))

        fnames = os.listdir(path)
        embeds = []
        topics = []
        n_updates = 0

        for filename in fnames:
            # process text files only
            if not(filename.endswith('.txt')):
                embeds.append(math.nan)
                topics.append('NaN')
                continue

            # don't repeat calculation if the file has already been processed 
            has_tags = len(existing_df['tags'][existing_df["file_names"]==filename])>=1
            
            try:
                has_embeds = len(existing_df['embeddings'][existing_df["file_names"]==filename].to_list()[0]) >5 
            except:
                has_embeds = False
        
            f = open(os.path.join(path,filename), 'r')
            text = f.read()
            if count_tokens(text)>4096:
                # keep 2000 words only
                text = " ".join(text.split()[0:2000])

            if has_tags:
                tags= existing_df['tags'][existing_df["file_names"]==filename].to_list()[0]
                topics.append(tags)
            else:
                n_updates +=1
                print(f'> > {filename}: extracting topics')
                done= False
                while not(done):
                    try:    
                        tags = self.extract_tags(text)
                        done= True
                    except:
                        print('> > system overloaded, waiting 5 sec')
                        time.sleep(5)

                topics.append(tags)

            if has_embeds:
                embeds.append(existing_df['embeddings'][existing_df["file_names"]==filename].to_list()[0])
            else:
                n_updates +=1
                print(f'> > {filename}: processing embeddings')
                done = False
                while not(done):
                    try:            
                        embedding = self.compute_embeds(tags)
                        done= True
                    except:
                        print('> > system overloaded, waiting 5 sec')
                        time.sleep(5)
                embeds.append(embedding)

        df = pd.DataFrame({'file_names':fnames, 'tags':topics, 'embeddings':embeds})
        df.to_csv(os.path.join(path,'DATAFRAME.csv'), index=False)
        df = df.reset_index(drop=True)
        print(f"> > # UPDATES applied:{n_updates}")
        return df

    def extract_tags(self, text):
        text = text.split('user:')
        text = "".join(text[1:])

        chat = [{"role": "system", 
                    "content": "You recieve text and extract up to 10 different topic covered in the text. You output the topics separated by a comma (,)"}]
        chat.append({"role": "user", "content":f"extract tags:{text}"})
        API_response = openai.ChatCompletion.create(
                model=self.tldr_model,
                temperature=0,
                messages=chat)
        
        output = API_response['choices'][0]['message']['content']
        if ':' in output:
            output = output.split(':')
            output = "".join(output[1:])
        return output

    # ADD Free alternative (Huggingface Embeds)
    def compute_embeds(self, words):
        return openai.Embedding.create(input=words, engine=self.embed_model)['data'][0]['embedding']

    def DaVinci_tldr(self, text):
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=f"{text}\n\nTl;dr",
            temperature=0,
            max_tokens=200,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        return response['choices'][0]["text"]
    
    def tldr(self, text, to_language=None, with_model = ''):
        if self.tldr_model == 'gpt-3.5-turbo'or with_model=='gpt-3.5-turbo':
            text = text.replace('\n',' ')
            if to_language != None:
                context =f'tldr in {to_language}:'
                CHAT = [{"role": "system", "content":context},
                        {"role": "user", "content":f"'{text}'"}]
                
                response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            temperature=0,
                            max_tokens=200,
                            messages=CHAT)
                
                try:
                    return response['choices'][0]['message']['content']
                except:
                    pass

            else: 
                return self.DaVinci_tldr(text)
            
        
        if self.tldr_model == 'Vicuna' or with_model=='Vicuna':
            try:
                webui.set_text_gen_params(temperature=0.1)
                result = webui.oobabooga_textgen(prompt=f'Text Summarizer [Question]: summarize the following text: {text}\n[Answer]:')
                postprocessed = webui.post_process(result)
                return postprocessed
            except IndexError as e:
                return result
            except Exception as e:
                print(e)
                return ''
    
"""
OnlineSearchEngine:
to be implemented:
 - allows to extract content from the internet with http requests;
 - provide context to the VirtualAssistant
 - find a way to trigger online search
"""

class OnlineSearchEngine:
    # work in progress
    pass

"""
MISCELLANEOUS FUNCTIONS
"""
def count_tokens(vCountTokenStr):
    # Tokenize the input string
    blob = TextBlob(vCountTokenStr)
    tokens = blob.words

    # Count the number of tokens
    num_tokens = len(tokens)
    return num_tokens


def parse_conversation(string_chat):
    split1_chat = string_chat.split('user:')

    rebuilt = []

    for item in split1_chat:
        if 'system:' in item:
            rebuilt.append({"role":"system", "content":f"{item.split('ststem:')[-1]}"})
        if 'assistant:' in item:
            spl_item = item.split("assistant:")
            rebuilt.append({"role":"user", "content":f"{spl_item.pop(0)}"})
            
            while len(spl_item)>=1:
                rebuilt.append({"role":"assistant", "content":f"{spl_item.pop(0)}"})
    
    return rebuilt

def take_last_k_interactions(chat, max_tokens=4000):
    n_tokens = 0
    interactions = []

    for item in chat:
        n_tokens += count_tokens(item['content'])
        if n_tokens>= max_tokens:
            return interactions
        interactions.append(item)