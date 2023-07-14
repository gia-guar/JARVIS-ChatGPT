#!/usr/bin/env python3
import dotenv
dotenv.load_dotenv()
import re
import argparse
import os
from requests import Session
from typing import Generator, Union

import urllib3
urllib3.disable_warnings()

S2_API_KEY = os.environ['S2_API_KEY']


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


def download_pdf(session: Session, url: str, path: str, user_agent: str = 'requests/2.0.0'):
    # send a user-agent to avoid server error
    headers = {
        'user-agent': user_agent,
    }

    # stream the response to avoid downloading the entire file into memory
    with session.get(url, headers=headers, stream=True, verify=False) as response:
        # check if the request was successful
        response.raise_for_status()

        if response.headers['content-type'] != 'application/pdf':
            raise Exception('The response is not a pdf')

        with open(path, 'wb') as f:
            # write the response to the file, chunk_size bytes at a time
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)


def download_paper(session: Session, paper_id: str, directory: str = 'papers', user_agent: str = 'requests/2.0.0') -> Union[str, None]:
    try:
        directory = os.environ['workspace']
    except:
        pass
    
    paper = get_paper(session, paper_id, fields='paperId,title,isOpenAccess,openAccessPdf')

    # check if the paper is open access
    if not paper['isOpenAccess']:
        return None

    paperId: str =re.sub(r'\W+', '', paper['title']).encode("utf-8").decode("utf-8")
    pdf_url: str = paper['openAccessPdf']['url']
    pdf_path = os.path.join(directory, f'{paperId}.pdf')
    if os.path.isfile(pdf_path):
        return None

    # create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # check if the pdf has already been downloaded
    if not os.path.exists(pdf_path):
        download_pdf(session, pdf_url, pdf_path, user_agent=user_agent)

    return pdf_path


def download_papers(paper_ids: list[str], directory: str = 'papers', user_agent: str = 'requests/2.0.0') -> Generator[tuple[str, Union[str, None, Exception]], None, None]:
    # use a session to reuse the same TCP connection
    with Session() as session:
        for paper_id in paper_ids:
            try:
                yield paper_id, download_paper(session, paper_id, directory=directory, user_agent=user_agent)
            except Exception as e:
                yield paper_id, e


def main(args: argparse.Namespace) -> None:
    for paper_id, result in download_papers(args.paper_ids, directory=args.directory, user_agent=args.user_agent):
        if isinstance(result, Exception):
            return f"Failed to download '{paper_id}': {type(result).__name__}: {result}"
        elif result is None:
            return f"'{paper_id}' is not open access"
        else:
            return f"Downloaded '{paper_id}' to '{result}'"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', '-d', default='papers')
    parser.add_argument('--user-agent', '-u', default='requests/2.0.0')
    parser.add_argument('paper_ids', nargs='+', default=[])
    args = parser.parse_args()
    main(args)

def Main(paper_ids=[], dir='papers', user_agent = 'requests/2.0.0', ):
    outcome = ''
    if isinstance(paper_ids, str):
        if ',' in paper_ids: 
            paper_ids = paper_ids.split(',')
        else:
            paper_ids = paper_ids.split()
            
        paper_ids = (id.strip() for id in paper_ids)
    for paper_id, result in download_papers(paper_ids, directory=dir, user_agent=user_agent):
        if isinstance(result, Exception):
            outcome += f"Failed to download '{paper_id}': {type(result).__name__}: {result}\n"
        elif result is None:
            outcome += f"couldn't download '{paper_id} because it is not open access\n"
        else:
            outcome += f"{result}\n"
    return outcome