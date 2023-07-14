# AGENT
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.llms import OpenAI
from langchain.agents import Tool, AgentExecutor, ZeroShotAgent
from langchain.memory import ConversationBufferMemory 
from langchain.agents import initialize_agent, load_tools

from typing import Any
from Assistant.semantic_scholar.agent_tools import *
from Assistant.semantic_scholar.S2_tools import *
from langchain.agents import AgentType

class ResearchAssistant:
    def __init__(self, current_conversation, workspace = None, index_name = 'paperquestioning'):
        self.current_workspace = workspace
        self.index_name = index_name
        self.query_engine = None
        self.current_conversation = current_conversation

        self.ans = []
        self.docs = []

        if 'workspaces' not in os.listdir(os.getcwd()):
            os.mkdir('workspaces')
            print('\tinitializing Research Assistant but no workspace are available: begin a new search please')
        elif len(os.listdir('workspaces'))>0:          
            self.boot_workspace(workspace)

        print('\tResearch Assistant initialization done')      
        self.agent = generateResearchAgent(self, k=1)

    def boot_workspace(self, workspace):
        if os.path.isdir(workspace):
                print('\tinitializing Research Assistant with Workspace directory: ', workspace)

                # init vector store
                init_attempts = 0
                print(' ')
                self.docs = load_workspace(workspace)
                while True:
                    init_attempts +=1

                    try:
                        self.query_engine, self.Index = llama_query_engine(self.docs,pinecone_index_name=self.index_name)
                        break
                    except Exception as e:
                        print(f'initialization attempt {init_attempts} failed with exception {e}')
                        time.sleep(2)
                        if init_attempts<=3:continue
        else: return None

    # make wrappers to store results and info
    def wrapper_find_papers_from_query(self, query):  
        try:
            res = find_paper_from_query(query, 20)
            text = ''
            for result in res:
                if not result['isOpenAccess']:continue
                text += result['title']+'; paperId: '+result['paperId']
                self.ans.append( f"{result['title']}: {result['paperId']}")
            if len(text)==0: return "couldn't find any open access result"
            return text
        except Exception as e:
            return f'error: {e}'
        

    def load_pdf_to_pinecone(self, paths):
        # read the pdf
        if isinstance(paths, str): paths = [paths]
        if isinstance(paths, list): raise Exception

        for path in paths:
            if not path.endswith('.pdf'): continue
            content = readPDF(path)
            doc = Document(
                text = content,
                doc_id = uuid.uuid4().hex
            )
            self.docs.append(doc)

        # upload to Pinecone and synch index
        self.Index.insert(document=doc)
        # regresh the query engine
        self.query_engine = self.Index.as_query_engine()
        return

    def PROTOCOL_begin_new_workspace(self, query):
        # PRELIMINARY ASSESSMENT
        prompt_template = "Do you really need to search for something on internet?: {query} \n Answer Yes or No"
        llm = OpenAI(temperature=0)
        llm_chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate.from_template(prompt_template)
        )
        assessment =  llm_chain.predict(query = query)
        if 'yes' not in assessment.lower():
            return 'what should be the topic of the workspace?'
        
        # GOING WITH IT
        # preprocessing
        prompt_template = "Extract a search key from the following query: {query}"
        llm = OpenAI(temperature=0)
        llm_chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate.from_template(prompt_template)
        )
        # extraction of a query
        search_query = llm_chain.predict(query = query)

        # post processing of extracted search-query
        search_query = re.sub('[^0-9a-zA-Z]', ' ', search_query.lower())
        if "search key" in search_query: search_query=search_query.replace("search key","").strip()

        print('SEARCH QUERY: ', search_query)
        self.current_workspace = PaperSearchAndDownload(query=search_query)
        self.boot_workspace(self.current_workspace)
        return f'new workspace created at {self.current_workspace}'
    
    def wrapper_download_paper(self, id):
        if 'cache' not in os.listdir(self.current_workspace): os.mkdir(os.path.join(self.current_workspace,'cache'))
        ans = download_pdf_from_id(paperid= id, path= os.path.join(self.current_workspace, 'cache'))
        update_workspace_dataframe(self.current_workspace, verbose = False)
        pdf_paths = ans.split('\n')
        self.load_pdf_to_pinecone(pdf_paths)
        return ans
    
    def wrapper_find_reccomendation(self, paperId):
        return find_recommendations(paper=paperId, result_limit=5)
    
    def find_in_papers(self, query):
        attempt =0
        while True:
            try: 
                attempt +=1
                answer = self.query_engine.query(query)
            except Exception as e:
                if attempt<=3: continue
                return str(e)
            return answer
        


# generate a Zero Shot React Agent with memory that looks K interactions behind
def generateResearchAgent(RA:ResearchAssistant, k:int):

    findpapers = Tool(
        name='Find from query',
        description='find a paper from a query, title and/or other information.',
        func= RA.wrapper_find_papers_from_query
    )

    download_ID = Tool(
        name='Download ID',
        description='download a paper from paperId. Take as input a paperId',
        func=RA.wrapper_download_paper
    )


    peek = Tool(
        name='glimpse pdf',
        description="get paper information if available, Take as input a paper title",
        func=glimpse_pdf
    )

    reccomend = Tool(
        name='find reccomendations',
        description='find similar paper from paperId. Take as input a paperId',
        func=RA.wrapper_find_reccomendation
    )
    
    tools = [findpapers, download_ID, peek, reccomend]
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, human_prefix='user', ai_prefix='assistant')

    # need to work on a custom LangChain llm model 
    prefix = """You are an assistant designed to browse scientific libraries, you have the following tools to complete the user requests:"""
    suffix = """Begin!"

    {chat_history}
    Question: {input}
    {agent_scratchpad}"""

    prompt = ZeroShotAgent.create_prompt(
        tools, 
        prefix=prefix, 
        suffix=suffix, 
        input_variables=["input", "chat_history", "agent_scratchpad"]
    )
    llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)

    # adding a window of memory:
    memory = build_memory(chat_history = RA.current_conversation(), k=k)
    
    return AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory, early_stopping_method ='generate', max_iterations=20)

def build_memory(chat_history, k):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, human_prefix='user', ai_prefix='assistant')
    k = min(k, len(chat_history)//2)

    if k==0 :return memory

    if chat_history[-k]["role"] != 'user': 
        print('refreshing memory warning - considering last interaction only')
        k=1
    try:
        for i in range(-k*2-1, -1, 2):
            input  = chat_history[i]['content']
            output = chat_history[i+1]['content']
            memory.save_context({"input":input}, {"output":output})
    except:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, human_prefix='user', ai_prefix='assistant')
    return memory
