# JARVIS-ChatGPT: A convesational assistant equipped with J.A.R.V.I.S's voice
**A voice-based interactive assistant equipped with a variety of synthetic voices (including J.A.R.V.I.S's voice from IronMan)**
<p align="center">
  <img src="https://user-images.githubusercontent.com/49094051/224847586-75810675-c4ad-4bbe-87e0-9c375b8a8aa0.PNG"/>

</p>

Ever dreamed to ask a hyper intelligent system tips to improve your armor? Now you can! Well, maybe not the aromor part... This project exploits OpenAI Whisper, OpenAI ChatGPT and IBM Watson.
* *[new] ++ International Language supported (see tutorial downside) ++*

AIM: *many times ideas come in the worst moment and they fade away before you have the time to explore them better. The objective of this project is developping a system capable of giving tips and opinions in quasi-real-time about anything you ask. The ultimate assistant will be able to be accessed from any authorized microphone inside your house or from your phone, it should run constantly in the background and when summoned will be able to generate meaningful answers (with a badass voice) as well as interfacing with the pc or a server and save/read/write files that can be accessed later.*

What you'll need:

 - An OpenAI account 
 - ffmpeg
 - python virtual enviroment (my venv runs on python 3.7 in case you'd need this info)
 - Some credit to spend on chatGPT (you can get three months of free usage by making signing up to OpenAI) (optional)
 - An OpenAI API key (optional, but strongly suggested)
 - An IBM Cloud account to exploit their cloud-based text-to speech models (tutorial: https://www.youtube.com/watch?v=A9_0OgW1LZU) 
 - mic and speaker (if you have many microphones you might be reuired to tell which audio you plan to use in the `get_audio.py`) 
 - CUDA capable grpahic engine (my Torch Version: 1.12.1+cu113, CUDA v11.2)
 

## ChatGPT
The easiest way to get answers from ChatGPT is to connect with to the service via cloud. To do this you can adopt 2 strategies:
 1) Using unofficial chatgpt-wrapper: someone amazing made this wrapper to have an ongoing conversation on ChatGPT from Command line or from your python script (https://github.com/mmabrouk/chatgpt-wrapper)
 2) Using your API key you'll be able to send and recieve stuff from the DaVinci003 model (the one that powers ChatGPT itself) [what we are going to do]
 
 3) using unofficial SDK: it's a further option that should be viable (https://github.com/labteral/chatgpt-python)

Option 2 is the most straightfoward from March 2023 since the latest OpenAI API support chats. However, you need to have some sort of credit on your account (wither paid or got for free). This option is implemented in the `openai_api_chatbot.py` script;
Option 1 was my earlier choice: it uses a wrapper to connect to your chatGPT account so you need to authenticate manually every time and follow instructions on the authors'github. [WARNING] you maight be exposed to fails due to server traffic limitations unless you are subscribed to a premium plan (see more at [ChatGPT Plus](https://openai.com/blog/chatgpt-plus) )
You'll find this option on `openai_wrapper_chatbot.py` but it's not being updated any longer. 

## The Idea:
Pretty straightfoward:

Microphone > pyaudio > audio.wav   
audio.wav > OpenAI Whisper > prompt
prompt > OpenAI ChatGPT > answer(text)  
answer *(text)* > IBM Watson/TTS-model > answer *(spoken)*

## March 13 2023 UPDATE: JARVIS VOICE IS HERE!
**How i did it**: I spent a huge amount of time on @CorentinJ github https://github.com/CorentinJ/Real-Time-Voice-Cloning which provides an interface to generate audio from text using a pretrained text-to-speech model. The GUI is pretty clever and I admire his work, however, using the model in a python script is not straight foward! I first edited the toolbox to save **embeddings**, which are the beginning of  the generation process,. They are the "voice ID" of the targeted people, expressed in terms of matrix. With this edit, I used the toolbox to generate Paul Bettany's voice embedding. <br>
Then, I wrote down a trimmed version of CorentinJ's toolbox, `JARVIS.py`. This version can load the embedding learned from Jarvis voice and do basic oprations like Synth and vocode upon request from any script. 

![toolbox](https://user-images.githubusercontent.com/49094051/224836993-ee7b4964-e518-46f4-85b1-b25f48f1a78c.PNG)
<p align="center"> Original Toolbox interface: you can see the embedding </p>

# TUTORIAL
## Github overview
**MAIN** script you should run: `openai_api_chatbot.py` if you want to use the latest version of the OpenAI API. If you rely on the wrapper open `openai_wrapper_chatbot.py` instead. `da_vinci_demo.py` is a simple script that sends single prompts to chatgpt (no chat possible); you should verify the wrapper works properly with `chatgpt_wrapper.py` if you want to use the wrapper. The remaining scripts are supplementary to the voice generation.

## Step 1: installation, accounts, APIs... 
- verify your graphic engine and CUDA version are compatible with pytorch by running `torch.cuda.is_available()` and `torch.cuda.get_device_name(0)`; 
- get the OpenAI API from their officila website. this will power Whisper and ChatGPT. 
- Authorize yorself by copying-pasting the API key inside `openai.api_key = 'your key'` (edit these code lines on the **MAIN** script with your key);
- get a IBM account up and running by following the youtube video (it require a credit card but there is a service that allows limited usage free of charge);
- copy-paste the url and the api key when authorizing and setting up the cloud service inside the __main__() function of the principal script;
- [WARNING] if you get errors try to run demos ( *_demo.py) to see if the probelm is with openai/wrapper. In case: check `pip openai --version`, if the problem is with the wrapper, check if you followed the instructions at the author's github and try to run `chatgpt install` with an open chrome tab; this got me some troubles at first as well.


## Step 2: Language suppport
- To speak with chatgpt in any language you should check if your language is supported by the speech generator at __https://cloud.ibm.com/docs/text-to-speech?topic=text-to-speech-voices__; 
- add or change the `voice` variable inside the `say()` function (e.g. for japanese add an entry *'ja':'ja-JP_EmiV3Voice'* for italian  *'it':'it-IT_FrancescaV3Voice'* is already added since i am italian);
![lang](https://user-images.githubusercontent.com/49094051/224839783-85ee6733-53d3-4d11-845c-5eb10c10c3f3.PNG)
- remember: The loaded Whisper is the medium one. If it performs badly in your language, upgrade to the large one in the _ _main_ _() at `whisper_model = whisper.load_model("large")`;

## Step 3: Running (`openai_api_chatbot.py`):
when running, you'll see much information being displayed.
- When *Star recording* is prompted, ask a quetion with a microphone close by;
- when you're done press CTRL+C (once);
- the script will convert the recorded audio to text using Whisper;
- the script will then expand the `chat_history` with your question and will update it as soon as it recieve a full answer from ChatGPT;
- if Hey Jarvis has been said `say()` will generate a waveform using Jarvis's embedding. 
- elsewise submit the taks to IBM Text-To-Speech services;
- when the stop key words are said the script will ask chatgpt to give a title to the conversation and will save the chat in a .txt file with the format 'Current-Date-Title.txt'

![Capture](https://user-images.githubusercontent.com/49094051/224842933-9d9bcdb2-8483-496c-a083-775ecdaa18aa.PNG)
![Capture2](https://user-images.githubusercontent.com/49094051/224842418-1caa61c5-a0a7-45ed-a563-e1bbde1c204e.PNG)

<br>

- to stop just say 'OK THANKS'
- To summon JARVIS voice just say 'HEY JARVIS' at some point;


# History:
- [x] [11 - 2022] Deliver chat-like prompts from python from keyboard
- [x] [12 - 2022] Deliver chat-like prompts from python with voice
- [x] [2  - 2023] International language support for prompt and answers
- [x] [3  - 2023] Jarvis voice setted up
- [x] [3  - 2023] Save conversation

currently working on:
- [ ] Background execution & Voice Summoning
- [ ] International language support for voice commands
- [ ] Extend voice commands (make a beeter active assistant)

following:
- [ ] *project memory*: store chats, events, timelines and other relevant information for a given project to be accessed later by the user or the assistant itself 
- [ ] Improve output displayed info
- [ ] Improve JARVIS voice performaces though propmpt preprocessing
- [ ] Create a full stack VirtualAssistant class with memory and local storage access
- [ ] Add sound feedback of different stages (chimes, beeps...)

### waiting for ChatGPT4 to:
- [ ] add multimodal input (i.e. "do you think 'this' [holding a paper plane] could fly" -> camera -> ChatGPT4 -> "you should improve the tip of the wings" )
- [ ] Extend *project memory* to images


Have fun!

if you have questions contact me at gianmarco.guarnier@hotmail.com
