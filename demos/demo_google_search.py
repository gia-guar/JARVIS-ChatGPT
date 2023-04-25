from langchain.utilities import GoogleSerperAPIWrapper
from dotenv import load_dotenv
load_dotenv()
from langchain.memory import ConversationBufferMemory
from langchain.llms.openai import OpenAI
from langchain import OpenAI, LLMChain 
from langchain.agents import initialize_agent, Tool, load_tools
from langchain.agents import AgentType, ZeroShotAgent, AgentExecutor

import geocoder
from newsapi import NewsApiClient
import os



def generateGoogleAgent():
    names = ['wikipedia', 'requests', 'open-meteo-api']
    
    if len(os.getenv('SERPER_API_KEY'))>1:
        print('(using google-seper)')
        names.append('google-serper')
    elif len(os.getenv('GOOGLE_API_KEY'))>1:
        print('(using google-serch)')
        names.append('google-search')
    
    tools = load_tools(names)
    custom_tools = [
        Tool(
            name ='Locate me',
            func = locate_me, 
            description='useful to know the current geographical location'),
        Tool(
            name='News',
            func=news,
            description='Use this when you want to get information about the top headlines of current news stories. The input should be a keyword describing the topic')
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

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, human_prefix='user', ai_prefix='assistant')
    
    return AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory, early_stopping_method = 'generate', max_iterations=2)


def news(keyword):
    newsapi = NewsApiClient(api_key=os.getenv('NEWS_API_KEY'))
    top_headlines = newsapi.get_top_headlines(q=keyword)
    if len(top_headlines['articles'])==0:
        top_headlines = newsapi.get_everything(q=keyword)
        top_headlines['articles'] = top_headlines['articles'][0:min(len(top_headlines['articles']),10)]

    res = ''

    for article in top_headlines['articles']:
        res += '\n'+article['title']+'\nurl: '+article['url']+'\n'

    print(res)
    return res

def locate_me(p):
    g = geocoder.ip('me')
    loc = str(g.city)+', '+ str(g.state)+', '+str(g.country) 
    print(loc)
    return loc

llm = OpenAI(temperature=0)

names = ["requests"]


tools = load_tools(names)
tools.append(Tool(
        name='Locate Me',
        func=locate_me,
        description='Helpful when you need to access the current geographical postition',
        ))

tools.append(Tool(
    name='News',
    func=news,
    description='Use this when you want to get information about the top headlines of current news stories. The input should be a keyword describing the topic'
))


self_ask_with_search = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
self_ask_with_search = generateGoogleAgent()
print(self_ask_with_search.run("What are the latest news about AI?"))