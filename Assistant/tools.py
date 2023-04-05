import openai
import os
import pandas as pd
import numpy as np
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity
import regex as re
import langid
from textblob import TextBlob
from tqdm import tqdm
import translators as ts

"""
Translator: 
performs basic translation opration using ChatGPT. 
Setting temperature to 0 allows better raw results
"""

"""
options:
 - gpt-3.5-turbo
 - translators 5.6.3 lib
"""
class Translator:
    def __init__(self, model="translators", ):
        self.DEFAULT_CHAT = [{"role": "system", 
                    "content": "You are a translator. You recieve text and target language as inputs and translate the text to the target language"}]
        self.body = None
        self.model = model

    def translate(self, input, to_language):
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
            res = ts.translate_text(input, translator='google',to_language='en')
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
                 translator_model = "translators",
                 default_dir = os.path.realpath(os.path.join(os.getcwd(),'saved_chats')),
                 irrelevancy_th=0.8):
        
        self.translate_engine = Translator(model=translator_model)
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
                print('dataset not updated. Updating it now...')
                self.produce_folder_tags()
        
        print('Analyzing DataFrame:')

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
            if count_tokens(text)>4096:
                # keep 2000 words only
                text = " ".join(text.split()[0:2000])

            tags = self.extract_tags(text)
            topics.append(tags)
            embeds.append(openai.Embedding.create(input=tags, engine=self.embed_model)['data'][0]['embedding'])
        
        df = pd.DataFrame({'file_names':fnames, 'tags':topics, 'embeddings':embeds})
        df.to_csv(os.path.join(path,'DATAFRAME.csv'), index=False)
        df = df.reset_index(drop=True)
        print(df.head())
        return df
    

    def extract_query(self, prompt):
        """
        Strategy: version 1: 
         1 - create a prompt embedding (encode meaning)
         2 - for every word, compare it's embedding to the prompt embedding, keep only the ones that are not noise
         3 - identify the words that are meaningful to the prompt

         Strategy: ideal: use a Multi Head Self Attention mechanisms
         This is not accessible through OpenAI API so it'll need a model trained from scratch.
         Then Multi head self attention can be used to provide weights to each word.
        """

        prompt_embed = np.array(openai.Embedding.create(input=prompt, engine='text-embedding-ada-002')['data'][0]['embedding'])
        negative_embedding = np.array(openai.Embedding.create(input='conversation, discussion', engine='text-embedding-ada-002')['data'][0]['embedding'])

        # init 
        results = []
        words = []
        embeds = []
        df = pd.DataFrame()

        # replacing all non alpha-numeric stuff with spaces
        # this is to avoid "word" and "word." to be considered different
        prompt = re.sub('[^0-9a-zA-Z]+', ' ', prompt)

        for word in prompt.split():
            if word.lower() in words: continue

            word_embedding = np.array(openai.Embedding.create(input=word, engine='text-embedding-ada-002')['data'][0]['embedding'])
            
            # if the word is not relevant to a search query
            if cosine_similarity(word_embedding, negative_embedding)>0.85:
                continue
            
            results.append(cosine_similarity(word_embedding, prompt_embed))
            words.append(word.lower())
            embeds.append(prompt_embed)

        # if no word is meaningful enough:
        if max(results)<self.irrelevancy_threshold:
            # ask ChatGPT to produce search keys:
            # to be implemented
            tags = self.gpt_extract_tags(prompt)
            return tags

        results = [float(i)/max(results) for i in results]

        df = pd.DataFrame({'words':words, 'similarity':results, 'embeddings':embeds})

        # find the keywords (research tags)
        df = df.sort_values(by='similarity', ascending=False)
        keys = []

        for i in range(len(df['similarity'])):
            word = df['words'][i]
            score = df['similarity'][i]
            
            #  if relative 'power' [%] is high enough
            if score >=0.95: 
                keys.append(word)

        print(f"Keys extracted: {keys}")
        return keys
        

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
    
    def gpt_extract_tags(self, text):
        # to be implemented
        return 

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