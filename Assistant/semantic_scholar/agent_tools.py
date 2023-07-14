from contextlib import contextmanager
import uuid
import os
import tiktoken

from . import S2_tools as scholar

import csv
import sys
import requests

# pdf loader
from langchain.document_loaders import OnlinePDFLoader

## paper questioning tools
from llama_index import Document
from llama_index.vector_stores import PineconeVectorStore
from llama_index import GPTVectorStoreIndex, StorageContext, ServiceContext
from llama_index.embeddings.openai import OpenAIEmbedding
 

def PaperSearchAndDownload(query):
    # make new workspace 
    if not os.path.exists( os.path.join(os.getcwd(),'workspaces') ): os.mkdir(os.path.join(os.getcwd(),'workspaces'))
    workspace_dir_name = os.path.join(os.getcwd(),'workspaces',query.split()[0] + '_'+ str(uuid.uuid4().hex))
    os.mkdir(workspace_dir_name)
    os.mkdir(os.path.join(workspace_dir_name,'results'))
    os.mkdir(os.path.join(workspace_dir_name,'refy_suggestions'))
    os.environ['workspace'] = workspace_dir_name

    # 1) search papers
    print('  1) Searching base papers') 
    papers = scholar.find_paper_from_query(query, result_limit=10)
    if len(papers == 0):
        papers = scholar.find_paper_from_query(query, result_limit=50)
    scholar.update_dataframe(incomplete=papers, dest=os.path.join(workspace_dir_name, 'results','papers.csv'))
    delete_duplicates_from_csv(csv_file=os.path.join(workspace_dir_name, 'results','papers.csv'))

    # 2) Cross-reference reccomendation system:
    # a paper is reccomended if and only if it's related to more than one paper
    print('\n\n 2) Expanding with Scholar reccomendations')
    counts = {}
    candidates = {}
    for paper in papers:
        guesses = scholar.find_recommendations(paper)

        for guess in guesses:
            if not guess['isOpenAccess']: continue

            candidates[guess['title']] = guess
            if guess['title'] not in counts.keys(): counts[guess['title']] = 1
            else: counts[guess['title']] += 1
    
    # reccomend only papers that appeared more than once 
    reccomends = []
    for key in counts:
        if counts[key]>1: reccomends.append(candidates[key])

    print(f'found {len(reccomends)} additional papers')
    # update the csv
    scholar.update_dataframe(incomplete= reccomends, dest=os.path.join(workspace_dir_name, 'results','papers.csv'))
    delete_duplicates_from_csv(csv_file=os.path.join(workspace_dir_name, 'results','papers.csv'))

    # download the papers (1/2)
    print('downloading papers (1/2)')
    with open(os.path.join(workspace_dir_name,'results','papers.csv'), 'r',encoding='utf-8') as fp:
        csvfile = csv.DictReader(fp)  
        scholar.download_pdf_from_id(" ".join( row['paperId'] for row in csvfile), workspace_dir_name)
    
    scholar.write_bib_file(csv_file=os.path.join(workspace_dir_name,'results','papers.csv'), bib_file=os.path.join(workspace_dir_name,'results','papers.bib'))

    # expand further with refy reccomendendation system
    print('\n\n 3) Expanding with Refy reccomendendation system')
    print('this might take a while...')
    scholar.refy_reccomend(bib_path=os.path.join(workspace_dir_name,'results','papers.bib'))

    with open(os.path.join(workspace_dir_name, 'refy_suggestions', 'test.csv'), 'r',encoding='utf-8') as fp:
        csvfile = csv.DictReader(fp) 
        for row in csvfile:
            title = scholar.replace_non_alphanumeric(row['title'])
            title = title.replace(" ","_")

            save_path = os.path.join(workspace_dir_name,'refy_suggestions',(title+'.pdf'))
            try:
                download_paper(url=row['url'], save_path=save_path)
            except:
                print(f'couldn t download {row}')

    return f'{os.path.join(os.getcwd(), workspace_dir_name)}'


import urllib
def download_paper(url, save_path=f"{uuid.uuid4().hex}.pdf"):
    success_string = f"paper saved successfully at {os.path.join(os.path.abspath(save_path))}"
    if url.endswith('.pdf'):
        urllib.request.urlretrieve(url, save_path)
        return success_string
    if 'doi' in url:
        doi = paper_id = "/".join(url.split("/")[-2:])
        # Construct the Crossref API URL
        print(doi)
        doi_url = f"https://doi.org/{doi}"

        # Send a GET request to the doi.org URL
        response = requests.get(doi_url, allow_redirects=True)

        # Check if the request was successful
        if response.status_code == 200:
            # Extract the final URL after redirection
            url = response.url

    if 'arxiv' in url:
        # URL del paper su arXiv

        # Ottieni l'ID del paper dall'URL
        paper_id = url.split("/")[-1]

        # Costruisci l'URL di download del paper
        pdf_url = f"http://arxiv.org/pdf/{paper_id}.pdf"

        # Scarica il paper in formato PDF
        urllib.request.urlretrieve(pdf_url, save_path)
        return success_string

    else:
        if '/full' in url:
            urllib.request.urlretrieve(url.replace('/full','/pdf'))
            return success_string
        if 'plos.org' in url:
            final_url = url.replace('article?', 'article/file?')
            urllib.request.urlretrieve(final_url, save_path)
            return success_string
    
    return f'\nfailed to download {url}'
        

def download_bibtex_library(csv_path):
    with open(csv_path, 'r',encoding='utf-8') as fp:
        csvfile = csv.DictReader(fp) 
        for row in csvfile:
            title = scholar.replace_non_alphanumeric(row['title'])
            title = title.replace(" ","-")

            save_path = os.path.join(os.path.join(csv_path, '..', title+'.pdf'))
            try:
                download_paper(url=row['url'], save_path=save_path)
            except:
                try:
                    download_paper(url=row['url']+'.pdf', save_path=save_path)
                except:
                    print(f'couldn t download {row}')

def generate_chunks(text, CHUNK_LENGTH = 4000):
    enc = tiktoken.encoding_for_model("gpt-4")
    tokens = enc.encode(text)
    token_chunks = [tokens[i:i + CHUNK_LENGTH] for i in range(0, len(tokens), CHUNK_LENGTH)]

    word_chunks = [enc.decode(chunk) for chunk in token_chunks]
    return word_chunks


from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

import langid
import time

# def process_pdf_folder(folder_path):
#     if not os.path.exists(folder_path):
#         return 'the folder does not exist, check your spelling'

#     for item in os.listdir(folder_path):
#         if not item.endswith('.pdf'):continue
        
#         with open(os.path.join(folder_path,'SUMMARY.txt'), 'a', encoding='UTF-8') as write_file:
#             write_file.write(item)
#             write_file.write("\n\n\n")
#             txt = summarize_pdf(item, model='Vicuna')
#             try:
#                 write_file.write(txt)
#             except:
#                 print(txt)
    
#     with open(os.path.join(folder_path,'SUMMARY.txt'), 'r', encoding='UTF-8') as read_file:
#         return read_file.read()



# # def summarize_pdf(pdf_path, model= None):
#     text = readPDF(pdf_path)

#     # according to the TLDR Model, consider smaller chunks
#     text_chunks = generate_chunks(text, 700)

#     if model is not None:
#         summarizer = LocalSearchEngine(tldr_model=model)
    
#     summary=''
#     for chunk in text_chunks:
#         summary += summarizer.tldr(chunk)

#     return summary

def get_result_path(path, exclude = []):
    for item in os.listdir(path):
        if item == 'papers.csv':
            return os.path.join(path, item)
        if os.path.isdir(os.path.join(path, item)) and item not in exclude: 
            res = get_result_path(os.path.join(path, item))
            if res: return res
    return 

def get_workspace_titles(workspace_name):
    csv_file_path = get_result_path(workspace_name)
    papers_available = []
    with open(csv_file_path, 'r', encoding='utf-8') as file:
        csv_file = csv.DictReader(file)
        for row in csv_file:
            papers_available.append(row['title'])
    return papers_available

import re
def same_title(title1, title2):
    try:
        title1 = re.sub(r'[^a-zA-Z]', ' ', title1)
        title2 = re.sub(r'[^a-zA-Z]', ' ', title2)
    except:
        return False
    words1 = set(title1.lower().split())
    words2 = set(title2.lower().split())
    return words1 == words2 or words1 <= words2 or words1 >= words2


def glimpse_pdf(title):
    # find papers.csv in workspace
    
    for workspace_name in os.listdir('workspaces'):
        csv_file_path = get_result_path(workspace_name)
        if csv_file_path is None: return 'no paper found'
        
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            csv_file = csv.DictReader(file)
            for row in csv_file:
                if same_title(row['title'], title): return f"{row['title']}, paperId: {row['paperId']}, summary: {row['abstract']}"
    
    return f'\nno paper found with title {title}'
    
def count_tokens(text):
    enc = tiktoken.encoding_for_model("gpt-4")
    tokens = enc.encode(text)
    return len(tokens)

def readPDF(pdf_path):
    loader = OnlinePDFLoader(pdf_path)
    data  = loader.load()
    text_content = ''

    for page in data:
        formatted_content = page.page_content.replace('\n\n', ' ')
        text_content+=formatted_content
    
    return text_content

def get_pdf_path(dir, exclude=[]):
    paths = []
    for item in os.listdir(dir):
        itempath = os.path.join(dir,item)
        if item.endswith('.pdf'): paths.append(itempath)
        if os.path.isdir(itempath)and item not in exclude: 
            subpaths = get_pdf_path(itempath)
            for i in subpaths: paths.append(i)
    return paths

def delete_duplicates_from_csv(csv_file):
    print('verifying duplicates...') 
    to_delete = []
    def delete_csv_row_by_title(csv_file, title):
        # Read the CSV file and store rows in a list
        with open(csv_file, 'r',encoding='UTF-8') as file:
            reader = csv.DictReader(file)
            rows = list(reader)

        # Find the row index with the matching title
        row_index = None
        for index, row in enumerate(rows):
            if row['title'] == title:
                row_index = index
                break

        # If no matching title is found, return
        if row_index is None:
            print(f"No row with title '{title}' found.")
            return

        # Remove the row from the list
        del rows[row_index]

        # Write the updated rows back to the CSV file
        with open(csv_file, 'w', newline='',encoding='UTF-8') as file:
            fieldnames = reader.fieldnames
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    
    
    with open(csv_file, 'r', encoding='UTF-8') as file:
        DELETED = 0
        reader = csv.DictReader(file)
        rows = list(reader)
        entries = set()
        for row in rows:
            if row['title']=='' or row['title'] is None: continue
            if row['title'] not in entries:entries.add(row['title'])
            else:
                DELETED+=1
                to_delete.append(row['title'])
                
        for title in to_delete: delete_csv_row_by_title(csv_file, title=title)
    print(f"Deleted {DELETED} duplicates")
    return
            
        

def update_workspace_dataframe(workspace, verbose = True):
    ADDED = 0
    # find results.csv 
    csv_path = get_result_path(workspace)

    # get titles in csv
    titles = get_workspace_titles(workspace)

    # get local papers path
    paths = get_pdf_path(workspace, exclude='refy_suggestions')

    # adding new to csv:
    for path in paths:
        
        exists = False

        # extract the title from the local paper
        title = scholar.extract_title(path)

        for t in titles:
            if same_title(t,title): exists = True
            
        
        # add it to dataframe if it was not found on the DF
        if not exists:
            if verbose: print(f"\nnew paper detected: {title}")
            # find it with online
            paper = scholar.find_paper_online(path)
            if paper : 
                if verbose: print(f"\t---> best match found online: {paper['title']} " )
                for t in titles:
                    if same_title(paper['title'], title): 
                        if verbose: print(f"\t    this paper is already present in the dataframe. skipping")
            else: 
                if verbose: print(path,   '-x-> no match found')
                continue
            with open(csv_path, 'a', encoding='utf-8') as fp:
                areYouSure = True
                for t in titles:
                    if same_title(t,paper['title']): areYouSure =False
                if not areYouSure:
                    if verbose: print(f"double check revealed that the paper is already in the dataframe. Skipping") 
                    continue
                if verbose: print(f"\t---> adding {paper['title']}")
                ADDED +=1
                paper_authors = paper.get('authors', [])
                journal_data = {}
                if 'journal' in paper:
                    journal_data = paper.get('journal',[])
                if journal_data is not None:
                    if 'name' not in journal_data: journal_data['name'] = ''
                    if 'pages' not in journal_data: journal_data['pages'] = ''

                if paper.get('tldr',[]) != []:tldr = paper['tldr']['text']
                elif paper.get('summary',[]) != []:tldr = paper['summary']
                elif 'abstract' in paper:tldr = paper['abstract']
                else: tldr = 'No summary available'

                if 'year' in paper:
                    year = paper['year']
                elif 'updated' in paper:year = paper['updated']
                else: year = ''

                if 'citationStyles' in paper:
                    if 'bibtex' in paper['citationStyles']: citStyle = paper['citationStyles']['bibtex'] 
                    else: citStyle = paper['citationStyles'][0]
                else: citStyle = ''

                csvfile = csv.DictWriter(fp, ['paperId', 'title', 'first_author', 'year', 'abstract','tldr','bibtex','influentialCitationCount','venue','journal','pages'])
                try:
                    csvfile.writerow({
                        'title': paper['title'],
                        'first_author': paper_authors[0]['name'] if paper_authors else '',
                        'year': year,
                        'abstract': paper['abstract'] if 'abstract' in paper else '',
                        'paperId': paper['paperId'] if 'paperId' in paper else '',
                        'tldr':tldr,
                        'bibtex':citStyle,
                        'influentialCitationCount': paper['influentialCitationCount'] if 'influentialCitationCount' in paper else '0',
                        'venue':paper['venue'] if 'venue' in paper else '',
                        'journal':journal_data['name'] if journal_data is not None else '',
                        'pages':journal_data['pages'] if journal_data is not None else '',
                    })      
                except Exception as e: 
                    if verbose: print('could not add ', title, '\n',e)
                # delete dupes if present

    if verbose: print(f"\n\nCSV UPDATE: Added {ADDED} new papers")
    
    # clean form dupes
    delete_duplicates_from_csv(csv_path)
    
    # update bib
    scholar.write_bib_file(csv_path)
    return
    


def load_workspace(folderdir):
    docs =[]
    
    for item in os.listdir(folderdir):
        if item.endswith('.pdf'):
            print(f'   > loading {item}')
            with suppress_stdout():
                content = readPDF(os.path.join(folderdir, item))
                docs.append(Document(
                    text = content,
                    doc_id = uuid.uuid4().hex
                ))
        
        if item =='.'or item =='..':continue
        if os.path.isdir( os.path.join(folderdir,item) ):
            sub_docs = load_workspace(os.path.join(folderdir,item))
            for doc in sub_docs:
                docs.append(doc)
        
    return docs

# List paths of all pdf files in a folder
def list_workspace_elements(folderdir):
    docs =[] 
    for item in os.listdir(folderdir):
        if item.endswith('.pdf'):
            docs.append(rf"{os.path.join(folderdir,item)}")
        
        if item =='.'or item =='..':continue
        if os.path.isdir( os.path.join(folderdir,item) ):
            sub_docs = list_workspace_elements(os.path.join(folderdir,item))
            for doc in sub_docs:
                docs.append(doc)
    return docs

def llama_query_engine(docs:list, pinecone_index_name:str):
    pinecone.init(
        api_key= os.environ['PINECONE_API_KEY'],
        environment= os.environ['PINECONE_API_ENV']
    )
    
    # Find the pinecone index
    if pinecone_index_name not in pinecone.list_indexes():
        # we create a new index
        pinecone.create_index(
            name=pinecone_index_name,
            metric='dotproduct',
            dimension=1536  # 1536 dim of text-embedding-ada-002
        )

    index = pinecone.Index(pinecone_index_name)
    
    # init it
    vector_store = PineconeVectorStore(pinecone_index=index)
    time.sleep(1)

    # setup our storage (vector db)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    embed_model = OpenAIEmbedding(model='text-embedding-ada-002', embed_batch_size=100)
    service_context = ServiceContext.from_defaults(embed_model=embed_model)

    
    # populate the vector store
    LamaIndex = GPTVectorStoreIndex.from_documents(
        docs, storage_context=storage_context,
        service_context=service_context
    )

    print('PINECONE Vector Index initialized:\n',index.describe_index_stats())

    # init the query engine
    query_engine = LamaIndex.as_query_engine()
    
    return query_engine, LamaIndex



@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout