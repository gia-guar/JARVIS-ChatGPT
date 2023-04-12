import openai
import os
import pandas as pd
import numpy as np
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity
from tqdm import tqdm

import regex as re
import langid
from textblob import TextBlob
import translators as ts
import argostranslate.package
import argostranslate.translate

import math
import time 

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

        # Download and install Argos Translate packages
        argostranslate.package.update_package_index()
        available_packages = argostranslate.package.get_available_packages()
        langs = kwargs['translator_languages']
        for i in range(len(langs)):
            for j in range(len(langs)):
                if langs[i]==langs[j]: continue
                print(f'downloading/verifying Argos Translate Language package from {langs[i]} to {langs[j]}...')
                package_to_install = next(
                    filter(
                        lambda x: x.from_code == langs[i] and x.to_code == langs[j], available_packages
                    )
                )
                argostranslate.package.install_from_path(package_to_install.download())
        

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
            res = ts.translate_text(input, translator='google', to_language=to_language, from_language=from_language)
            return res
        
        if self.model == 'argostranslator':
            res = argostranslate.translate.translate(input, from_code=from_language, to_code=to_language)
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
                 translator_model = "argostranslators",
                 translator_languages = ['en','it','es'],
                 default_dir = os.path.realpath(os.path.join(os.getcwd(),'saved_chats')),
                 irrelevancy_th=0.8):
        
        self.translate_engine = Translator(model=translator_model, translator_languages=translator_languages)
        self.tldr_model  = tldr_model
        self.embed_model = embed_model
        self.default_dir = default_dir
        self.irrelevancy_threshold = irrelevancy_th
    
    def compute_similarity(self, key, text):
        key_embedding = openai.Embedding.create(input=key, engine=self.embed_model)['data'][0]['embedding']
        query_embedding = openai.Embedding.create(input=text, engine=self.embed_model)['data'][0]['embedding']
        similarity = cosine_similarity(key_embedding, query_embedding)
        return similarity
    

    def accurate_search(self, key, path=None, n=-1, from_csv=False):
        if path is None:
            path = self.default_dir

        if isinstance(key, list) or isinstance(key, tuple):
            key = " ".join(key)
        
        # USE EXISTING DATAFRAME TO MAKE SEARCH FASTER (skip tag generation)
        if from_csv:
            DataFrame = pd.read_csv(os.path.join(path,'DATAFRAME.csv'))
            fnames = DataFrame["file_names"]
            tags = DataFrame['tags']

            if len(fnames)!=len(os.listdir(path)):
                print('> dataset not updated. Updating it now...')
                self.produce_folder_tags()
        
        print('> Analyzing DataFrame:')

        results = []
        topics = []

        for i in tqdm(range(len(fnames))):
            if not(fnames[i].endswith('.txt')):
                results.append(0)
                topics.append('None')
                continue

            file_tags = tags[i]
            topics.append(file_tags)

            if langid.classify(file_tags)[0] != langid.classify(key)[0]:
                file_tags = self.translate_engine.translate(input=file_tags, to_language=langid.classify(key)[0])

            relevance = self.compute_similarity(file_tags, key)
            results.append(relevance)

        if n==-1: n=len(fnames)
        df = pd.DataFrame({'file_names':fnames, 'similarity':results,"tags":topics})
        df = df.sort_values(by='similarity', ascending=False)
        df = df.reset_index(drop=True)

        return df.head(n)

    

    def quick_search(self, key, path=None, model="text-davinci-003", temp = 0, n=-1):    
        if path is None:
            path = self.default_dir
        
        fnames = os.listdir(path)
        results = []

        for filename in fnames:
            if not(filename.endswith('txt')):
                results.append(None)
                continue

            f = open(os.path.join(path,filename), 'r')
            text = f.read()
            if count_tokens(text)>4096:
                # keep 2000 words only
                text = " ".join(text.split()[0:2000])

            prompt=(f"is the following text related with {key}? Text:{text}.\nanswer yes or no.")
            completions = openai.Completion.create(
                engine=model,
                prompt=prompt,
                max_tokens=1024,
                temperature=temp)
            
            if 'yes' in completions.choices[0].text.upper():
                results.append(True)
            else:
                results.append(False)
        
        df = pd.DataFrame({'file_names':fnames, 'similarity':results})
        df = df.sort_values(by='similarity', ascending=False)
        
        if n==-1: n=len(fnames)
        return df
    

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

    def compute_embeds(self, words):
        return openai.Embedding.create(input=words, engine=self.embed_model)['data'][0]['embedding']

    def tldr(self, text, to_language):
        context =f'tldr in {to_language}:'
        CHAT = [{"role": "system", "content":context},
                {"role": "user", "content":f"'{text}'"}]

        response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    temperature=0,
                    messages=CHAT)
        
        return response['choices'][0]['message']['content']
    
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


def count_tokens(vCountTokenStr):
    # Tokenize the input string
    blob = TextBlob(vCountTokenStr)
    tokens = blob.words

    # Count the number of tokens
    num_tokens = len(tokens)
    return num_tokens