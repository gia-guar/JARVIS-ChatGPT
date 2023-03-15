# new conversation is starting
bot = ChatGPT()
i=0
while True:
    i+=1
    question = str(input(str(i)+'> '))
    response = bot.ask(question)
    print(response)

    if 'ok thanks' in question: 
        break