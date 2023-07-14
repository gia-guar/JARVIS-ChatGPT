import json 
import requests
import re
import langid

SERVER = 'localhost'
TEXT_GEN_PARAMS = {
    'max_new_tokens': 200,
    'do_sample': True,
    'temperature': 0.72,
    'top_p': 0.73,
    'typical_p': 1,
    'repetition_penalty': 1.1,
    'encoder_repetition_penalty': 1.0,
    'top_k': 0,
    'min_length': 0,
    'no_repeat_ngram_size': 0,
    'num_beams': 1,
    'penalty_alpha': 0,
    'length_penalty': 1,
    'early_stopping': False,
    'seed': -1,
    'add_bos_token': True,
    'custom_stopping_strings': [],
    'truncation_length': 2048,
    'ban_eos_token': False,
}

def set_text_gen_params(**kwargs):
    for key in kwargs:
        if key not in list(TEXT_GEN_PARAMS.keys()): raise Exception('no such parameter in oogabooga text generation')
        TEXT_GEN_PARAMS[key]=kwargs[key]


def oobabooga_textgen(prompt, params=TEXT_GEN_PARAMS, server=SERVER):
    ChatMode = True if type(prompt) == list else False
    
    if ChatMode: 
        nMessages = len(prompt)
        prompt = parse_conversation(prompt) 

    payload = json.dumps([prompt, params])
    APIresponse = requests.post(f"http://{server}:7860/run/textgen", json={
        "data": [
            payload
        ]
    }).json()
    reply = APIresponse["data"][0]
    
    # hallucination filter:
    if ChatMode:
        reply = reply.replace("[assistant]:","###")
        reply = reply.replace("[user]:","###")
        reply = reply.replace("[system]:","###")
        reply = reply.split('###')
        reply = " ".join(reply[(nMessages+1):(nMessages+2)])
    
    return reply
    
def post_process(answer):
    allowed = ['Answer','Outcome','Discussion','Conclusion']
    answer = answer.split('[Question]')[-1]

    relevant =''
    for a in allowed:
        if a in answer:
            temp = re.split(r'\[|\]', answer)
            try:
                relevant += temp[temp.index(a)+1].strip(':')
            except:
                print('Failure processing answer')
                pass
        
    print(len(relevant.split()))
    return relevant
    
def parse_conversation(chat):
    linkDetectionRegexStr = "[a-zA-Z0-9]((?i) dot |(?i) dotcom|(?i)dotcom|(?i)dotcom |\.|\. | \.| \. |\,)[a-zA-Z]*((?i) slash |(?i) slash|(?i)slash |(?i)slash|\/|\/ | \/| \/ ).+[a-zA-Z0-9]"
    oobaboogaChatHistory = ""
    for message in chat:
        oobaboogaChatHistory += f"[{str(message['role'])}]:{message['content']}\n"
    oobaboogaChatHistory = re.sub(linkDetectionRegexStr, "<url>", oobaboogaChatHistory)
    return oobaboogaChatHistory