import openai

## SET YOUR API KEY:
openai.apikey('')

def generate_single_response(prompt, model_engine="text-davinci-003", temp=0.5):
    openai.api_key = 'your openai api key'  
    prompt = (f"{prompt}")
    completions = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=1024,
        temperature=temp
    )
    return completions.choices[0].text

def get_answer(message):
    CHAT = [{"role": "system", "content": """You are a prompt manager. 
    You have access to a suite of actions and decide if a prompt require one of them to be executed. Your actions:
    1 - find a file: file can store past conversations, textual information;
    2 - respond: the user wants to have a general answer to a topic;
    3 - adjust settings;
    4 - find a directory inside the Computer;
    5 - save this conversation for the future
    6 - check on internet
    7 - None of the above
    
    You can answer only with numbers. A number must always be present in your answer"""},{"role":"user", "content":"PROMPT: "+message}]

    response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                temperature=0,
                max_tokens=1,
                messages=CHAT)
    
    return response



if __name__ == "__main__":
    
    while True:
        message = input("[user]: ")
        answer = get_answer(message=message)
        print(answer['choices'][0]['message']['content'])