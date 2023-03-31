import openai
import os
import pandas as pd
import numpy as np
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity
import regex as re
import langid
"""
Translator: 
performs basic translation opration using ChatGPT. 
Setting temperature to 0 allows better raw results
"""
class Translator:
    def __init__(self, model="gpt-3.5-turbo"):
        self.DEFAULT_CHAT = [{"role": "system", 
                    "content": "You are a translator. You recieve text and target language as inputs and translate the text to the target language"}]
        self.body = None
        self.model = model

    def translate(self, input, language):
        self.body = self.DEFAULT_CHAT
        self.body.append({"role":"user", "content":f"translate in {language}:'{input}'"})
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
    



"""
LocalSearchEngine:
 - Looks for files in a foder;
 - extracts information;
 - create high value contents that allow for accurate search;

to be implemented:
 - extend reserarch to .pdf and .jpeg (w/ ChatGPT4)
    - extends also to videos;
    - extends also to scientific papers;
 - itegrate communication with VirtualAssistant class; 
 - integrate voice commands;

"""

class LocalSearchEngine:
    def __init__(self, 
                 embed_model = "text-embedding-ada-002", 
                 tldr_model = "gpt-3.5-turbo",
                 translator_model = "gpt-3.5-turbo",
                 default_dir = os.path.realpath(os.path.join(os.getcwd(),'saved_chats'))):
        
        self.translate_engine = Translator(model=translator_model)
        self.tldr_model  = tldr_model
        self.embed_model = embed_model
        self.default_dir = default_dir
    
    def compute_similarity(self, key, text):
        key_embedding = openai.Embedding.create(input=key, engine=self.embed_model)['data'][0]['embedding']
        query_embedding = openai.Embedding.create(input=text, engine=self.embed_model)['data'][0]['embedding']
        similarity = cosine_similarity(key_embedding, query_embedding)
        return similarity
    
    def accurate_search(self, key, path=None, n=-1, from_csv=False):
        if path is None:
            path = self.default_dir

        # USE EXISTING DATAFRAME TO MAKE SEARCH FASTER (skip tag generation)

        try:
            if from_csv:
                DataFrame = pd.read_csv(os.path.join(path,'DATAFRAME.csv'))
                fnames = DataFrame["file names"]
                tags = DataFrame['tags']

                if len(fnames)!=len(os.listdir(path)):
                    print('dataset not updated')
                    raise Exception
                
                print('Using existing DataFrame')

                results = []
                for i in range(len(fnames)):
                    if not(fnames[i].endswith('.txt')):
                        results.append(0)
                        continue

                    file_tags = tags[i]

                    if langid.classify(file_tags) != langid.classify(key):
                        file_tags = self.translate_engine.translate(input=file_tags, language=langid.classify(key))

                    relevance = self.compute_similarity(file_tags, key)
                    results.append( relevance)

                if n==-1: n=len(fnames)
                df = pd.DataFrame({'file names':fnames, 'similarity':results})
                df = df.sort_values(by='similarity', ascending=False)

                return df.head(n)

        except Exception as e:
            print(e)
            pass

        print('\nBuilding temporary DataFrame. This may take a while')
        fnames = os.listdir(path)
        results = []
        topics = []

        for filename in fnames:
            if not(filename.endswith('.txt')):
                results.append(0)
                continue
            f = open(os.path.join(path,filename), 'r')
            text = f.read()
            tags = self.extract_tags(text)
            topics.append(tags)

            if langid.classify(tags) != langid.classify(key):
                tags = self.translate_engine.translate(input=tags, language=langid.classify(key))

            relevance = self.compute_similarity(tags,key)
            results.append( relevance)

        if n==-1: n=len(fnames)
        df = pd.DataFrame({'file names':fnames, 'similarity':results})
        df = df.sort_values(by='similarity', ascending=False)

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
        
        df = pd.DataFrame({'file names':fnames, 'similarity':results})
        df = df.sort_values(by='similarity', ascending=False)
        
        if n==-1: n=len(fnames)
        return df
    




    def produce_folder_tags(self, path=None, update=False):
        if path is None:
            path = self.default_dir
        
        if ('DATAFRAME.csv' in os.listdir(path)) and update==False:
            print('DataFrame already existing')
            return
        
        fnames = os.listdir(path)
        embeds = []
        topics = []

        for filename in fnames:
            if not(filename.endswith('.txt')):
                embeds.append(None)
                topics.append(None)
                continue

            f = open(os.path.join(path,filename), 'r')
            text = f.read()
            tags = self.extract_tags(text)
            topics.append(tags)
            embeds.append(openai.Embedding.create(input=tags, engine=self.embed_model)['data'][0]['embedding'])
        
        df = pd.DataFrame({'file names':fnames, 'tags':topics, 'embeddings':embeds})
        df.to_csv(os.path.join(path,'DATAFRAME.csv'), index=False)
        print(df.head())
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

        # post processing (when output is like "topics: ...")
        if ':' in output:
            output = output.split(':')
            output = "".join(output[1:])
        return output
    


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