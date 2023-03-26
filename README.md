# JARVIS-ChatGPT: A convesational assistant equipped with J.A.R.V.I.S's voice
**A voice-based interactive assistant equipped with a variety of synthetic voices (including J.A.R.V.I.S's voice from IronMan)**
<p align="center">
  <img src="https://user-images.githubusercontent.com/49094051/227788148-a8ff8e06-86a4-41a6-aa53-8b7d6855360c.png"/>
  <span style=color:grey> <i>image by MidJourney AI </i> </span>
</p>

Ever dreamed to ask a hyper intelligent system tips to improve your armor? Now you can! Well, maybe not the aromor part... This project exploits OpenAI Whisper, OpenAI ChatGPT and IBM Watson.

---
## March 26 2026 UPDATE: Background execution and Hands Free control
This update is focused on making the assistant more viable in everyday life. 
 - You can now follow some instructions you can find at <span style="color:green"> BootJARVIS.pdf </span> to run the main script automatically when the system boots;
 - There's no more need to press Ctrl+C to deliver mic recordings to Whisper. To read about hows *Hands Free* and *Summoning* work, go check the tutorial below or the update history (UpdateHistory.md);
 - minor improvements:
    1. Adding the package pytts3x for text to speech when IBM is un-available (monthly usage expired);
    2. improved CLI outputs;
    3. improved descriptions;
---
<p align="center"> <strong> PROJECT MOTIVATION:  </strong> </p> 

*Many times ideas come in the worst moment and they fade away before you have the time to explore them better. The objective of this project is developping a system capable of giving tips and opinions in quasi-real-time about anything you ask. The ultimate assistant will be able to be accessed from any authorized microphone inside your house or from your phone, it should run constantly in the background and when summoned should be able to generate meaningful answers (with a badass voice) as well as interfacing with the pc or a server and save/read/write files that can be accessed later. In addition, it might interface with some external gadgets (IoT) but that's extra.*

## What you'll need:

 - An [OpenAI](https://openai.com) account 
 - [ffmpeg](https://ffmpeg.org/) 
 - python virtual enviroment (my venv runs on python 3.7, requirements.txt are compatible with this version only)
 - Some credit to spend on chatGPT (you can get three months of free usage by making signing up to OpenAI) (strognly suggested)
 - An OpenAI API key (strongly suggested)
 - An IBM Cloud account to exploit their cloud-based text-to speech models (tutorial: https://www.youtube.com/watch?v=A9_0OgW1LZU) (optional)
 - mic and speaker (if you have many microphones you might be reuired to tell which audio you plan to use in the `get_audio.py`) 
 - CUDA capable graphic engine (my Torch Version: 1.12.1+cu113, CUDA v11.2)
 

## Connecting with ChatGPT: 3 ways
The easiest way to get answers from ChatGPT is to connect to the service via cloud using an API. To do this you can adopt 2 strategies:
 1) Using unofficial chatgpt-wrapper: someone amazing made this wrapper to have an ongoing conversation on ChatGPT from Command line or from your python script (https://github.com/mmabrouk/chatgpt-wrapper)
 2) Using your OpenAI API key you'll be able to send and recieve stuff from the DaVinci003 model (the one that powers ChatGPT itself) [what we are going to do]
 3) using unofficial SDK: it's a further option that should be viable (https://github.com/labteral/chatgpt-python)

Option 2 is the most straightfoward from March 2023 since the latest OpenAI API support chats. However, you need to have some sort of credit on your account (wether paid or got for free when subscribing). This option is implemented in the `openai_api_chatbot.py` script;
Option 1 was my earlier choice: it uses a wrapper to connect to your chatGPT account so you need to authenticate manually every time and follow instructions on the authors'github. [WARNING] you might be exposed to fails due to server traffic limitations unless you are subscribed to a premium plan (see more at [ChatGPT Plus](https://openai.com/blog/chatgpt-plus) )
You'll find this option implemented at `openai_wrapper_chatbot.py` but it's not being updated any longer. 




# TUTORIAL
## GitHub overview
**MAIN** script you should run: `openai_api_chatbot.py` if you want to use the latest version of the OpenAI API. If you rely on the wrapper open `openai_wrapper_chatbot.py` instead. `da_vinci_demo.py` is a simple script that sends single prompts to chatgpt (no chat possible); you should verify the wrapper works properly with `chatgpt_wrapper_demo.py` if you want to use the wrapper. `get_audio.py` stores all the functions to handle mic interactions.<br>The remaining scripts are supplementary to the voice generation and should not be edited.

## Step 1: installation, accounts, APIs... 
- Verify your graphic engine and CUDA version are compatible with pytorch by running `torch.cuda.is_available()` and `torch.cuda.get_device_name(0)`; 
- Get the OpenAI API from their official website. This will allow to send and recieve material to Whisper and ChatGPT. 
- Authorize yorself by copying-pasting the API key inside `openai.api_key = 'your key'` (edit these code lines on the **MAIN** script with your key);
- Get a IBM Cloud account up and running by following the youtube video (it will require a credit card at some point but there is a service that allows limited usage free of charge);
- Copy-paste the url and the api key when authorizing and setting up the cloud service inside the __main__() function of the principal script;
- [WARNING] If you get errors try to run demos ( *_demo.py) to see if the probelm is with openai/wrapper. In case: check `pip openai --version`; if the problem is with the wrapper, check if you followed the instructions at the author's github and try to run `chatgpt install` with an open chrome tab; this got me some trouble at first as well.


## Step 2: Language suppport
- To have answers spoken in your language you should first check if your language is supported by the speech generator at __https://cloud.ibm.com/docs/text-to-speech?topic=text-to-speech-voices__; 
- If it's supported, add or change the `voice` variable inside the `say()` function (e.g. for japanese add an entry *'ja':'ja-JP_EmiV3Voice'*, italian  *'it':'it-IT_FrancescaV3Voice'* is already added since I am italian);<br>

<p align="center">
  <img src="https://user-images.githubusercontent.com/49094051/224839783-85ee6733-53d3-4d11-845c-5eb10c10c3f3.PNG"/>
</p>

- Remember: The loaded Whisper is the medium one. If it performs badly in your language, upgrade to the larger one in the ```__main__()``` at `whisper_model = whisper.load_model("large")`; but I hope your GPU memory is large likewise.

## Step 3: Running (`openai_api_chatbot.py`):
when running, you'll see much information being displayed. I'm costantly striving to improve the readability of the execution, this is still a beta. Anyway, this is what happens when you hit 'run':
- Preliminary initializations take place;
- When *awaiting for triggering words* is displayed you'll need to say `ELEPHANT` to summon the assistant. This magic word can be switched, but it needs to be english. At this point a conversation will begin and you can speak in whatever language you want (if you followed step 2). The conversation will terminate when you say a [stop word](https://github.com/gianmarcoguarnier/JARVIS-ChatGPT/edit/main/README.md#key-words) or when you stop making question for more than 30 seconds (still unstable, needs to be imrpved) <br>
<p align="center">
  <img src="https://user-images.githubusercontent.com/49094051/227788246-c85bc84c-396f-4e45-9a37-ff9857b0c770.PNG" /><br>
</p>

- The word *listening...* should then appear. At this point you can make your question. When you are done just wait (3 seconds) for the answer;
- The script will convert the recorded audio to text using Whisper;
- The script will then expand the `chat_history` with your question it will send a request with the API an it will update the history as soon as it recieves a full answer from ChatGPT (this may take up to 5-10 seconds, consider explicitly asking for a short answer if you are in a hurry);
- If 'Hey Jarvis' has been said, the `say()` function will use the voice-duplicating toolbox to generate a waveform using Jarvis's voice embedding;
<p align="center">
 <img src='https://user-images.githubusercontent.com/49094051/227788211-4257f2e4-8aef-48f4-aae6-174c7ff5007a.PNG'/><br>
  <i>you can ignore the error</i>
</p>

- Elsewise, the taks is submitted to IBM Text-To-Speech services or pyttsx3;
- When any of stop key words are said the script will ask chatgpt to give a title to the conversation and will save the chat in a .txt file with the format 'CurrentDate-Title.txt';
- The assistant will then go back to sleep;
<p align="center">
 <img src='https://user-images.githubusercontent.com/49094051/227788180-b9da0957-a58b-4c1c-bc34-4a4c8a0e0957.PNG'/><br>
  <i>i made some other prompt, ignore the title mentioning healthcare</i>
</p>


# Key words:
- to stop or save the chat, just say 'OK THANKS' at some point;
- To summon JARVIS voice just say 'HEY JARVIS' at some point;

<span style="color:grey">*not ideal i know but works for now*</span>


# History:
- [x] [11 - 2022] Deliver chat-like prompts from python from keyboard
- [x] [12 - 2022] Deliver chat-like prompts from python with voice
- [x] [2  - 2023] International language support for prompt and answers
- [x] [3  - 2023] Jarvis voice setted up
- [x] [3  - 2023] Save conversation
- [x] [3 - 2023] Background execution & Voice Summoning
- [x] [3 - 2023] Improve output displayed info
- [x] [3 - 2023] Improve JARVIS voice performaces though propmpt preprocessing

currently working on:
- [ ] International language support for voice commands
- [ ] Extend voice commands (make a beeter active assistant)

following:
- [ ] *project memory*: store chats, events, timelines and other relevant information for a given project to be accessed later by the user or the assistant itself 
- [ ] Create a full stack VirtualAssistant class with memory and local storage access
- [ ] Add sound feedback of different stages (chimes, beeps...)

### waiting for ChatGPT4 to:
- [ ] add multimodal input (i.e. "do you think 'this' [holding a paper plane] could fly" -> camera -> ChatGPT4 -> "you should improve the tip of the wings" )
- [ ] Extend *project memory* to images

<span style="color:grey">*Check the HistoryUpdate.md of the project for more insights.*</span>

Have fun!

if you have questions contact me at gianmarco.guarnier@hotmail.com
<p align="right"><i>Gianmarco Guarnier<i></p>
