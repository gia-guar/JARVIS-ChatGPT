# JARVIS-ChatGPT: A conversational assistant equipped with J.A.R.V.I.S's voice
**A voice-based interactive assistant equipped with a variety of synthetic voices (including J.A.R.V.I.S's voice from IronMan)**
<p align="center">
  <img src="https://user-images.githubusercontent.com/49094051/227788148-a8ff8e06-86a4-41a6-aa53-8b7d6855360c.png"/>
  <span style=color:grey> <i>image by MidJourney AI </i> </span>
</p>

Ever dreamed to ask hyper-intelligent system tips to improve your armor? Now you can! Well, maybe not the armor part... This project exploits OpenAI Whisper, OpenAI ChatGPT and IBM Watson.
<p align="center"> <strong> PROJECT MOTIVATION:  </strong> </p> 

*Many times ideas come in the worst moment and they fade away before you have the time to explore them better. The objective of this project is developing a system capable of giving tips and opinions in quasi-real-time about anything you ask. The ultimate assistant will be able to be accessed from any authorized microphone inside your house or your phone, it should run constantly in the background and when summoned should be able to generate meaningful answers (with a badass voice) as well as interface with the pc or a server and save/read/write files that can be accessed later. It should be able to run research, gather material from the internet (extract content from HTML pages, transcribe Youtube videos, find scientific papers...) and provide summaries that can be used as context to make informed decisions. In addition, it might interface with some external gadgets (IoT) but that's extra.*
<br>
---
## APRIL 5th 2023 UPDATE: New Voice Models (F.R.I.D.A.Y), Expanding Local Search Engine and More
I finally decided to upgrade the voice model from @ConrentinJ to [@CoquiAI](https://github.com/coqui-ai/tts). This model works on the same principle (Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis) but is much faster, more versatile and offers more options to explore. Right now, I just want to push this version live, it works by default with one of the models offered by the TTS package. In the future, I'll explore the differences and strengths of all the other models (you can do it by changing the name of the model when the Voice is initialized inside ``Voice.py``, as shown in the ``tts_demo.py``). Moreover, this model is multilanguage so if you find any clean interviews of voice actors you can use them as models when the answer needs to be spoken in your (or any) language.
<br>
Secondly, I've made some improvements to the Local Search Engine. Now it can be accessed with voice. In particular:
 1. Once you've made a prompt, ```LocalSearchEngine.analyze_prompt()``` will try to interpret the prompt and it will produce a flag, ranging from 1 to N (where N is the number of Actions the Assistant can make);
 2. If the flag corresponds to **"1"** the associated action will be **"Look for a file"** and that protocol will be triggered;
 3. The system will first communicate its intentions and if you confirm, the assistant will ask you to provide some search keywords;
 4. The system will utilize a pandas DataFrame, where some topic tags are associated to the conversation, to detect relevant discussions;
 5. Finally, the system will rank all the files from the most relevant to the least pertinent;
 6. The natural following step would be to recover one of the files, but this is still a work in progress;
<br>

Minor updates:
 - Bug fixes;
 - Added ``langid``, ``TextBlob`` and ``translators`` to get faster translations and reduce GPT credit usage;
 - Improved Speech-to-text by reducing the possible languages to the ones specified in the Assistant model;
 <br>
---

## What you'll need:
 - An [OpenAI](https://openai.com) account 
 - [ffmpeg] python virtual environment (my venv runs on python 3.7, requirements.txt are compatible with this version only)
 - Some credit to spend on ChatGPT (you can get three months of free usage by making signing up to OpenAI) (strongly suggested)
 - An OpenAI API key (strongly suggested)
 - An IBM Cloud account to exploit their cloud-based text-to-speech models (tutorial: https://www.youtube.com/watch?v=A9_0OgW1LZU) (optional);
 - A (reasonably) fast internet connection (most of the code relies on API so a slower connection might result in a longer time to respond)
 - mic and speaker (if you have many microphones you might be required to tell which audio you plan to use in the `get_audio.py`) 
 - CUDA capable graphic engine (my Torch Version: 1.12.1+cu113, CUDA v11.2 ```pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113```)
 - Patience :)

## Connecting with ChatGPT: 3 ways
The easiest way to get answers from ChatGPT is to connect to the service via cloud service using an API. To do this you can adopt 2 strategies:
 1) Using unofficial chatgpt-wrapper: someone amazing made this wrapper to have an ongoing conversation on ChatGPT from the Command line or from your Python script (https://github.com/mmabrouk/chatgpt-wrapper)
 2) Using your OpenAI API key you'll be able to send and receive stuff from the *DaVinci003 model* (the one that powers ChatGPT3) or from the *ChatGPT3.5 Turbo engine* [what we are going to do]
 3) using unofficial SDK: it's a further option that should be viable (https://github.com/labteral/chatgpt-python)

Option 2 is the most straightforward from March 2023 since the latest OpenAI API support chats. However, you need to have some sort of credit on your account (whether paid or got for free when subscribing). This option is implemented in the `openai_api_chatbot.py` script;
Option 1 was my earlier choice: it uses a wrapper to connect to your chatGPT account so you need to authenticate manually every time and follow instructions on the author's GitHub. It is a sub-optimal option because you can't have the system integrated at PC startup since it needs login. Moreover, you might be exposed to fails due to server traffic limitations unless you are subscribed to a premium plan (see more at [ChatGPT Plus](https://openai.com/blog/chatgpt-plus) )
You'll find this option implemented at `openai_wrapper_chatbot.py` but it's not being updated any longer. 




# TUTORIAL
## GitHub overview
**MAIN** script you should run: `openai_api_chatbot.py` if you want to use the latest version of the OpenAI API. If you rely on the wrapper open `openai_wrapper_chatbot.py` instead. Inside the deemos folder you'll find some guidance for the packages used in the project, if you have errors you might check these files first to target the problem. Mostly is stored in the Assistant folder: `get_audio.py` stores all the functions to handle mic interactions, `tools.py` implements some basic aspects of the Virtual Assistant, `voice.py` describes a (very) rough Voice class <br> The remaining scripts are supplementary to the voice generation and should not be edited.

## Step 1: installation, accounts, APIs... 
### Enviroment
1. Make a new, empty venv with Python 3.7;
2. ```pip install venv_requirements.txt```; This might take some time (~45 mins); if you encounter conflicts on specific packages, install them manually without the ```==<version>```;
3. install PyTorch: ```pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113``` (mind your CUDA version);
4. install [TTS](https://github.com/coqui-ai/tts);
5. download the Assistant and other scripts from this repo;
6. Check everything works *(following)*
<br>

### Checks
- Verify your graphic engine and CUDA version are compatible with PyTorch by running `torch.cuda.is_available()` and `torch.cuda.get_device_name(0)` inside Pyhton; 
- Get the OpenAI API key from their official website. This will allow us to send and receive material to Whisper and ChatGPT. 
- Authorize yourself by copying-pasting the API key inside `openai.api_key = 'your key'` (edit these code lines on the **MAIN** script with your key);
- Get an IBM Cloud account up and running by following the youtube video (it will require a credit card at some point but there is a service that allows limited usage free of charge);
- Copy-paste the URL and the API key when authorizing and setting up the cloud service inside the __main__() function of the principal script;
- [WARNING] If you get errors try to run demos ( *_demo.py) to see if the problem is with openai/wrapper. In case: check `pip openai --version`; if the problem is with the wrapper, check if you followed the instructions at the author's GitHub and try to run `chatgpt install` with an open Chrome tab; this got me some trouble at first as well.
- You can check the sources of error by running demos in the demos folder


## Step 2: Language support
- To have answers spoken in your language you should first check if your language is supported by the speech generator at __https://cloud.ibm.com/docs/text-to-speech?topic=text-to-speech-voices__; 
- If it's supported, add or change the languages inside ```VirtualAssistant.__init__()``` ;<br>

<p align="center">
  <img src="https://user-images.githubusercontent.com/49094051/224839783-85ee6733-53d3-4d11-845c-5eb10c10c3f3.PNG"/>
</p>

- Remember: The loaded Whisper is the medium one. If it performs badly in your language, upgrade to the larger one in the ```__main__()``` at `whisper_model = whisper.load_model("large")`; but I hope your GPU memory is large likewise.

## Step 3: Running (`openai_api_chatbot.py`):
When running, you'll see much information being displayed. I'm constantly striving to improve the readability of the execution, this is still a beta, forgive slight variations from the screens below. Anyway, this is what happens in general terms when you hit 'run':
- Preliminary initializations take place, you should hear a chime when the Assistant is ready;
- When *awaiting for triggering words* is displayed you'll need to say `ELEPHANT` to summon the assistant. This magic word can be switched, but it needs to be English. At this point, a conversation will begin and you can speak in whatever language you want (if you followed step 2). The conversation will terminate when you say a [stop word](https://github.com/gianmarcoguarnier/JARVIS-ChatGPT/tree/main#key-words) or when you stop making questions for more than 30 seconds (still unstable, needs to be improved) <br>
<p align="center">
  <img src="https://user-images.githubusercontent.com/49094051/227788246-c85bc84c-396f-4e45-9a37-ff9857b0c770.PNG" /><br>
</p>

- After the magic word is said, the word *listening...* should then appear. At this point, you can make your question. When you are done just wait (3 seconds) for the answer to be submitted;
- The script will convert the recorded audio to text using Whisper;
- The text will be analyzed and a decision will be made. If the Assistant believes you want to perform an action (like looking for a past conversation) the respective protocols will be initated; 
- Elsewise, the script will then expand the `chat_history` with your question it will send a request with the API and it will update the history as soon as it receives a full answer from ChatGPT (this may take up to 5-10 seconds, consider explicitly asking for a short answer if you are in a hurry);
- The `say()` function will perform voice duplication to speak with Jarvis/Someone's voice; if the argument is not in English, IBM Watson will send the response from one of their nice text-to-speech models. If everything fails, the functions will rely on pyttsx3 which is a fast yet not as cool alternative;
<p align="center">
 <img src='https://user-images.githubusercontent.com/49094051/227788211-4257f2e4-8aef-48f4-aae6-174c7ff5007a.PNG'/><br>
  <i>you can ignore the error</i>
</p>

- When any of the stop keywords are said, the script will ask ChatGPT to give a title to the conversation and will save the chat in a .txt file with the format 'CurrentDate-Title.txt';
- The assistant will then go back to sleep;
<p align="center">
 <img src='https://user-images.githubusercontent.com/49094051/227788180-b9da0957-a58b-4c1c-bc34-4a4c8a0e0957.PNG'/><br>
  <i><span style="color:grey">I made some other prompts, ignore the title mentioning healthcare</span> </i>
</p>


# Keywords:
- to stop or save the chat, just say 'THANKS' at some point;
- To summon JARVIS voice just say 'HEY JARVIS' at some point;

<span style="color:grey">*not ideal I know but works for now*</span>


# History:
- [x] [11 - 2022] Deliver chat-like prompts from Python from a keyboard
- [x] [12 - 2022] Deliver chat-like prompts from Python with voice
- [x] [2  - 2023] International language support for prompt and answers
- [x] [3  - 2023] Jarvis voice set up
- [x] [3  - 2023] Save conversation
- [x] [3  - 2023] Background execution & Voice Summoning
- [x] [3  - 2023] Improve output displayed info
- [x] [3  - 2023] Improve JARVIS's voice performances through prompt preprocessing
- [x] [4  - 2023] Introducing: *Project memory* store chats, events, timelines and other relevant information for a given project to be accessed later by the user or the assistant itself 
- [x][4  - 2023] Create a full stack ```VirtualAssistant``` class with memory and local storage access
- [x] Add sound feedback at different stages (chimes, beeps...)
- [x] International language support for voice commands (beta)

currently working on:
- [ ] fixing chat length bug (when the chat is too long it can't be processed by ChatGPT 3.5 Turbo)
- [ ] Extending voice commands and *Actions* (make a better active assistant)
- [ ] expanding *Memory*  

following:
- [ ] Include other NLP free models if ChatGPT is unavailable (my credit is about to end)
- [ ] Connect the system to the internet
- [ ] Refine memory and capabilities

### waiting for ChatGPT4 to:
- [ ] add multimodal input (i.e. "Do you think 'this' [holding a paper plane] could fly" -> camera -> ChatGPT4 -> "you should improve the tip of the wings" )
- [ ] Extend *project memory* to images, pdfs, papers...

<span style="color:grey">*Check the [UpdateHistory.md](https://github.com/gianmarcoguarnier/JARVIS-ChatGPT/blob/main/UpdateHistory.md) of the project for more insights.*</span>

Have fun!

if you have questions you can contact me at gianmarco.guarnier@hotmail.com
<p align="right"><i>Gianmarco Guarnier<i></p>
