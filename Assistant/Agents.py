from langchain import OpenAI, LLMChain
from langchain.llms import OpenAI
from langchain.agents import Tool, AgentExecutor, ZeroShotAgent
from langchain.memory import ConversationBufferMemory 
from langchain.agents import initialize_agent, load_tools
import datetime

from Assistant import VirtualAssistant
import os

# generate a Zero Shot React Agent with memory that looks K interactions behind
def generateReactAgent(VA:VirtualAssistant, k:int):
    # Local Search Engine ($)
    LocalSearchEngine = Tool(
        name= 'Key Search',
        func=VA.find_file,
        description="Useful when you don't know the name of a resource. Inputs should be keywords. Keywords are used to find resources. You ddon't know the name of the resources"
    )

    Save = Tool(name='Memorize', 
             func=VA.save_chat,
             description='save the current conversation. Useful for when the conversation will be needed in future')

    FileReader = Tool(
        name='Load File',
        func=VA.open_file,
        description='Useful when you have file names. Loads the content of a file given its filename'
    )
    
    Summarize = Tool(
        name='TLDR',
        func=VA.search_engine.tldr,
        description='Summarize large amounts of text'
    )
    tools = [LocalSearchEngine, Save, FileReader, Summarize]
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, human_prefix='user', ai_prefix='assistant')

    # need to work on a custom LangChain llm model 
    prefix = """You are an AI research assistant designed to assist users with their academic research. You are equipped with these tools:"""
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
    memory = build_memory(chat_history = VA.current_conversation(), k=k)
    
    return AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory, early_stopping_method ='generate', max_iterations=2)



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



def generateGoogleAgent(VA:VirtualAssistant, k:int):
    names = ['wikipedia', 'requests', 'open-meteo-api']
    llm = OpenAI(temperature=0)
    if len(os.getenv('SERPER_API_KEY'))>1:
        print('(using google-seper)')
        names.append('google-serper')
    elif len(os.getenv('GOOGLE_API_KEY'))>1:
        print('(using google-serch)')
        names.append('google-search')
    
    tools = load_tools(names, llm=llm)
    custom_tools = [
        Tool(
            name ='Locate me',
            func = locate_me, 
            description='useful to know the current geographical location'),
        Tool(
            name='News',
            func=news,
            description='Use this when you want to get information about the top headlines of current news stories. The input should be a keyword describing the topic'),
        Tool(
            name='Today',
            func=today,
            description='Useful to know the current day'),
        Tool(
            name='Delta days',
            func= time_between_dates,
            description='Use this you need to compute the time between two Dates. Input should be two dates in the ISO 8601 format: Year-Month-Day'
        )
        ]
    
    for item in custom_tools: tools.append(item)
        
    prefix = """Answer the question. You have also access to the following tools:"""
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
    memory = build_memory(VA.current_conversation(), k)
    
    return AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory, early_stopping_method = 'generate', max_iterations=4)



## FUNCTIONS FOR TOOLS
import geocoder
from newsapi import NewsApiClient

def locate_me(p):
    g = geocoder.ip('me')
    return [g.city, g.state, g.country]

def today(p):
    return str(datetime.date.today())

def time_between_dates(date1, date2):
    try:
        date1 = datetime.date.fromisoformat(date1)
        date2 = datetime.date.fromisoformat(date2)
    except:
        return 'date format incorrect'
    if date2.toordinal()>date1.toordinal(): return str(date2-date1)
    else: return str(date1-date2) 

def news(keyword):
    newsapi = NewsApiClient(api_key=os.getenv('NEWS_API_KEY'))
    top_headlines = newsapi.get_top_headlines(q=keyword)
    if len(top_headlines['articles'])==0:
        top_headlines = newsapi.get_everything(q=keyword)
        top_headlines['articles'] = top_headlines['articles'][0:min(len(top_headlines['articles']),10)]

    res = ''

    for article in top_headlines['articles']:
        res += '\nsource: '+article['source']['name']+'\n'
        res += '\n'+article['title']+'\nurl: '+article['url']+'\n'
        res += article['description']

    print(res)
    return res