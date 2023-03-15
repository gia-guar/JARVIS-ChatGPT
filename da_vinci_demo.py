import openai

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

if __name__ == "__main__":
    question = "Make a question here"
    answer = generate_single_response(question)
    print(answer)
