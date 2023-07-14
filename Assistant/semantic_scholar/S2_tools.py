import csv
import re
from time import time
import requests
import dotenv
import aspose.pdf as ap
dotenv.load_dotenv()

import argparse
import os
from requests import Session
from typing import Generator, Union
import subprocess
import urllib3
import json
urllib3.disable_warnings()

import refy

import pdftitle
from langchain.document_loaders import OnlinePDFLoader
import time
import arxiv
from pymed import PubMed

from .simple import Main

RESULT_LIMIT = 10
S2_API_KEY = os.environ['S2_API_KEY']

PAPER_FIELDS = 'paperId,externalIds,title,authors,year,abstract,isOpenAccess,openAccessPdf,influentialCitationCount,citationStyles,tldr,venue,journal'

def get_paper(session: Session, paper_id: str, fields: str = 'paperId,title', **kwargs) -> dict:
    params = {
        'fields': fields,
        **kwargs,
    }
    headers = {
        'x-api-key': S2_API_KEY,
    }

    with session.get(f'https://api.semanticscholar.org/graph/v1/paper/{paper_id}', params=params, headers=headers) as response:
        response.raise_for_status()
        return response.json()



def find_paper_from_query(query, result_limit=RESULT_LIMIT):
    papers = None
    while papers is None:
        try:
            while True:
                rsp = requests.get('https://api.semanticscholar.org/graph/v1/paper/search',
                                        params={'query': query, 'limit': result_limit, 'fields': PAPER_FIELDS})
                if rsp.status_code == 429: 
                    time.sleep(60)
                    continue
                break
            
            rsp.raise_for_status()
            results = rsp.json()
            total = results["total"]
            if not total:
                print('No matches found. Please try another query.')
                return 'No matches found. Please try another query.'

            

            papers = results['data']
            filtered = []
            for paper in (papers):
                if paper['isOpenAccess']: filtered.append(paper)
                if len(filtered)>=result_limit:break
                    
        except Exception as e:
            print('\n!!!!!!!!\n')
            print('ERROR OCCURRED: ',e)
            print(rsp)
            return rsp.status_code

    print(f'Found {total} results. OpenAccess: {len(filtered)}.')
    return papers

# Finds papers which are similar to an exisiting one
def find_recommendations(paper, result_limit = RESULT_LIMIT):

    print(f"Looking for up to {result_limit} recommendations based on: {paper['title']}")
    rsp = requests.get(f"https://api.semanticscholar.org/recommendations/v1/papers/forpaper/{paper['paperId']}",
                       params={'fields': 'title,url,isOpenAccess', 'limit': result_limit})
    
    rsp.raise_for_status()
    results = rsp.json()
    print_papers(results['recommendedPapers'])
    return results['recommendedPapers']

def extract_title(path):
    try:
        title = pdftitle.get_title_from_file(path)

        # remove non letter chars
        title = re.sub('[^0-9a-zA-Z]', ' ', title)

        # sometime the full text is returned instead of just the title. in that 
        # case use a coarse title detection
        if len(title.split())>30: raise Exception
        return title
    except:
        # ROUGH TITLE DETECTION
        loader = OnlinePDFLoader(path)
        data  = loader.load()
        text_content = ''
        formatted_content = data[0].page_content.replace('\n\n', ' ')
        text_content+=formatted_content
        
        # CONSIDER THE FIRST 100 WORDS
        # exclude single chars and remove puncts
        title = ' '.join([word for word in text_content.split()[0:min(100,len(text_content.split()))] if (len(word)>1)])
        title = re.sub('[^0-9a-zA-Z]', ' ', title)

        # option 1: take the capital words contained in the first 100 words
        title = ' '.join([word for word in title.split()[0:min(100,len(title.split()))] if (word.isupper() and len(word)>1)])
        if len(title)>20: title = " ".join(title.split()[0:10])

        # option 2: take the first 10 words 
        else:
            title = [word for word in text_content.split()[0:100] if (len(word)>1)]
            title = ' '.join( title[0:min(10,len(title))])

        print(' > generated title: ', title)
        if title=='': return []
        return title

def find_paper_online(path):
    # if every word of the original title is present in the result, return it
    def same_title(title1, title2):
        return (word in title2.lower().split() for word in title1.lower().split())
    
    # OPEN AND EXTRACT PAPER TITLE
    title = extract_title(path)

    # LOOK IN OTHER PAPER DATABASES
    # 1) scholar attempt
    while True:
        res = find_paper_from_query(title, result_limit=5)
        if isinstance(res, int):
            if res == 400: raise Exception
            if res == 429: 
                time.sleep(60)
            continue
        break
    if isinstance(res, list):
        for article in res:
            if same_title(title, article['title']): return article

    # 2) arxiv attempt
    search = arxiv.Search(
        query=title,
        id_list= [],
        max_results=5,
    )
    res = search.results()
    for article in res:
        if same_title(article.title, title): 
            return  article._raw
        
    # 3) pubmed attempt
    pubmed = PubMed(tool="MyTool", email="my@email.address")
    res = pubmed.query(title, max_results=5)
    
    for article in res:
        art = json.loads(article.toJSON())
        if same_title(title, art['title']):return art

    # noting found :(
    return 
    
def print_papers(papers):
    results = ''
    for idx, paper in enumerate(papers):
        results+= f"{idx}  {paper['title']} {paper['url']}"
    return results


def chunks(items, chunk_size):
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def fetch_paper_batch(paperid: list):
    req = {'ids': [f'{id}' for id in paperid]}
    # https://api.semanticscholar.org/api-docs/graph#tag/Paper-Data/operation/post_graph_get_papers
    
    rsp = requests.post('https://api.semanticscholar.org/graph/v1/paper/batch',
                        params={'fields': PAPER_FIELDS},
                        json=req)
    
    if rsp.status_code != 200:
        return f'Problem fetching {req}: ' + rsp.text
    return rsp.json()


def download_pdf_from_id(paperid, path=os.getcwd()):
    try:
        res = Main(paper_ids=paperid, dir=path)
        print(res)
        return res
    except:
        return f'error with {paperid}'

# add to PAPER.CSV semantic scolar entries from ID
def update_dataframe(incomplete, dest):
    results = fetch_paper_batch(paperid= [item['paperId'] for item in incomplete])
    if isinstance(results, str): 
        print(results) 
        print(f" input: {incomplete}")
        return
    pdf_des= dest
    pdf_des = pdf_des[:-4] + '.pdf'
    text = ''

    with open(pdf_des, 'a+',encoding='utf-8') as f:
        for paper in results:
            try:
                text += paper['title'].upper()+'\n'
                if 'tldr' in paper.keys():
                    if paper['tldr'] is not None:text += paper['tldr']['text']+'\n'
                text += paper['abstract']+'\n'
                if 'summary' in paper.keys(): text += paper['summary']+'\n'
                text += '\n\n'
            except:
                pass

    write_to_pdf(text, pdf_des)

    count = 0

    # Read existing entries from the CSV file
    existing_entries = set()

    isFile =  os.path.isfile(dest)
    if not isFile:
        with open(dest, 'w',encoding='utf-8') as fp:
            csvfile = csv.DictWriter(fp, ['paperId', 'title', 'first_author', 'year', 'abstract','tldr','bibtex','influentialCitationCount','venue','journal','pages'])
            csvfile.writeheader()
    if isFile:
        with open(dest, 'r',encoding='utf-8') as fp:
            csvfile = csv.DictReader(fp)
            for row in csvfile:
                existing_entries.add(row['paperId'])

    # Append new entries to the CSV file
    with open(dest, 'a', encoding='utf-8') as fp:
        csvfile = csv.DictWriter(fp, ['paperId', 'title', 'first_author', 'year', 'abstract','tldr','bibtex','influentialCitationCount','venue','journal','pages'])
        
        for paper in results:
            paperId = paper['paperId']
            if paperId in existing_entries:
                continue  # Skip if the entry already exists
            
            paper_authors = paper.get('authors', [])
            
            journal_data = {}
            if 'journal' in paper:
                journal_data = paper.get('journal',[])
            if journal_data is not None:
                if 'name' not in journal_data: journal_data['name'] = ''
                if 'pages' not in journal_data: journal_data['pages'] = ''

            if paper.get('tldt',[]) != []:
                tldr = paper['tldr']['text']
            elif paper.get('summary',[]) != []:
                tldr = paper['summary']
            else:
                tldr = paper['abstract']

            csvfile.writerow({
                'title': paper['title'],
                'first_author': paper_authors[0]['name'] if paper_authors else '',
                'year': paper['year'],
                'abstract': paper['abstract'],
                'paperId': paperId,
                'tldr':tldr,
                'bibtex':paper['citationStyles']['bibtex'] if paper['citationStyles']['bibtex'] else '',
                'influentialCitationCount':paper['influentialCitationCount'],
                'venue':paper['venue'],
                'journal':journal_data['name'] if journal_data is not None else '',
                'pages':journal_data['pages'] if journal_data is not None else '',
            })
            # except Exception as e:
            #     print('error adding paper: ',e, '\n',paper)
            #     paper['title']
            #     paper['year']
            #     paper['abstract']
            #     paper['citationStyles']['bibtex']
                
            #     if paper['tldr']: paper['tldr']
            #     if paper_authors: paper_authors[0]['name']

            #     quit()
            count += 1

    return f'Added {count} new results to {dest}'




def write_bib_file(csv_file, bib_file=None):
    if bib_file is None:
        bib_file = csv_file[:-4]+'.bib'
    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        with open(bib_file, 'w', encoding='utf-8') as output:
            print(f'writing bibtex file at {bib_file}')
            for row in reader:
                bib_entry = create_bib_entry(row)
                output.write(bib_entry + '\n\n')

def create_bib_entry(row):
    paper_id = row['paperId']
    title = row['title']
    author = row['first_author']
    year = row['year'].split('-')[0] # assuming format like 2023-03-24T15:46:10Z (arxiv use this)

    journal_match = re.search(r"journal\s*=\s*{([^}]*)}", row['bibtex'])
    if journal_match:
        journal = journal_match.group(1)
    else: journal = ''
    pages_match = re.search(r"pages\s*=\s*{([^}]*)}", row['bibtex'])
    if pages_match:
        pages = pages_match.group(1)
    else: pages = ''

    abstract = replace_non_alphanumeric(row['abstract'])

    # Generate the BibTeX entry
    bib_entry = f"@ARTICLE{{{paper_id},\n"
    bib_entry += f"  title     = \"{title}\",\n"
    bib_entry += f"  author    = \"{author}\",\n"
    bib_entry += f"  abstract  = \"{abstract}\",\n"
    bib_entry += f"  year      = {year},\n"
    bib_entry += f"  journal   = \"{journal}\",\n"
    bib_entry += f"  pages     = \"{pages}\"\n"
    bib_entry += "}"

    return bib_entry


def replace_non_alphanumeric(string, replacement=' '):
    pattern = r'[^a-zA-Z0-9]'
    replaced_string = re.sub(pattern, replacement, string)
    return replaced_string


def refy_reccomend(bib_path, number=20):
    d = refy.Recomender(
        bib_path,            # path to your .bib file
        n_days=30,               # fetch preprints from the last N days
        html_path=os.path.join(os.path.join(bib_path.replace('\\results\\papers.bib',''),'refy_suggestions'),"test.html"),   # save results to a .csv (Optional)
        N=number                 # number of recomended papers 
        )


def write_to_pdf(text, dest):
    # Initialize document object
    document = ap.Document()

    # Add page
    page = document.pages.add()

    # Initialize textfragment object
    text_fragment = ap.text.TextFragment(text)

    # Add text fragment to new page
    page.paragraphs.add(text_fragment)

    # Save updated PDF
    document.save(dest)