# JARVIS-ChatGPT: A conversational assistant equipped with J.A.R.V.I.S's voice
**A voice-based interactive assistant equipped with a variety of synthetic voices (including J.A.R.V.I.S's voice from IronMan)**
<p align="center">
  <img src="https://user-images.githubusercontent.com/49094051/227788148-a8ff8e06-86a4-41a6-aa53-8b7d6855360c.png"/>
  <span style=color:grey> <i>image by MidJourney AI </i> </span>
</p>

Ever dreamed to ask hyper-intelligent system tips to improve your armor? Now you can! Well, maybe not the armor part... This project exploits OpenAI Whisper, OpenAI ChatGPT and IBM Watson.
<p align="center"> <strong> PROJECT MOTIVATION:  </strong> </p> 

*Many times ideas come in the worst moment and they fade away before you have the time to explore them better. The objective of this project is to develop a system capable of giving tips and opinions in quasi-real-time about anything you ask. The ultimate assistant will be able to be accessed from any authorized microphone inside your house or your phone, it should run constantly in the background and when summoned should be able to generate meaningful answers (with a badass voice) as well as interface with the pc or a server and save/read/write files that can be accessed later. It should be able to run research, gather material from the internet (extract content from HTML pages, transcribe Youtube videos, find scientific papers...) and provide summaries that can be used as context to make informed decisions. In addition, it might interface with some external gadgets (IoT) but that's extra.*
<br>
<br>
<br>

<p align="center"> <strong> DEMO: </strong> </p> 

https://user-images.githubusercontent.com/49094051/231303323-9859e028-33e1-490d-9967-44852fd0efc5.mp4

<br>

---
## APRIL 11th 2023 UPDATE: Overall improvement to search engine, update README.md
new: ```pip install argostranslate pvporcupine python-dotenv```
- Upgrading to Python 3.8 and CUDA 11.7 (!)
- Lately, the ```translator``` package was taking too long to work (~20 seconds to get a translation), so I added another translator package that works instantly and it's offline;
- The 'Jarvis' wake-up keyword was added from the ```picovoice``` package. It requires a free key you can get at https://picovoice.ai/platform/porcupine;
- Fundamental improvements to the local search engine in terms of speed and credit consumption. With this update, accessing information from past conversations gets easier. When the search is completed the AI will summarize the text;
- Using dotenv for easier authentication; 
- Made a reasonable requirement file;
<br>
 
---

## What you'll need:
<p align="center"><i>DISCLAIMER:<br> The project might consume your OpenAI credit resulting in undesired billing;<br> I don't take responsibility for any unwanted charges;<br>Consider setting limitations on credit consumption at your OpenAI account; </i> </p> 

 - An [OpenAI](https://openai.com) account and API key;
 - [PicoVoice](https://picovoice.ai/platform/porcupine/) account and a free AccessKey; [optional]
 - [ffmpeg](https://ffmpeg.org/) ;
 - Python virtual environment (Python>=3.8 and <3.10);
 - Some credit to spend on ChatGPT (you can get three months of free usage by signing up to OpenAI);
 - CUDA version >= 11.2;
 - An IBM Cloud account to exploit their cloud-based text-to-speech models (tutorial: https://www.youtube.com/watch?v=A9_0OgW1LZU) (optional);
 - A (reasonably) fast internet connection (most of the code relies on API so a slower connection might result in a longer time to respond);
 - mic and speaker (if you have many microphones you might be required to tell which audio you plan to use in the `get_audio.py`);
 - CUDA capable graphic engine (my Torch Version: 2.0 and CUDA v11.7 ```pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117```);
 - Patience :sweat_smile:

You can follow a [YT video](https://www.youtube.com/watch?v=AcCG7DOwhG8&t=1s) I made to guide you through the installation, but remember it's getting obsolete quickly so mind this README for specifications like the Python version (3.8 instead of 3.7)

# TUTORIAL
## GitHub overview
**MAIN** script you should run: `openai_api_chatbot.py` if you want to use the latest version of the OpenAI API Inside the demos folder you'll find some guidance for the packages used in the project, if you have errors you might check these files first to target the problem. Mostly is stored in the Assistant folder: `get_audio.py` stores all the functions to handle mic interactions, `tools.py` implements some basic aspects of the Virtual Assistant, `voice.py` describes a (very) rough Voice class <br> The remaining scripts are supplementary to the voice generation and should not be edited.

## Step 1: installation, accounts, APIs... 
### Environment
1. Make a new, empty venv with Python 3.8 and activate it (.\venv_name\Scripts\activate );
2. ```pip install -r venv_requirements.txt```; This might take some time; if you encounter conflicts on specific packages, install them manually without the ```==<version>```;
3. install manually PyTorch according to your CUDA VERSION;
5. install [TTS](https://github.com/coqui-ai/tts);
6. Run [their script](https://github.com/coqui-ai/TTS/blob/dev/README.md#-python-api) and check everything is working (it should download some models) (you can alternatively run ```demos/tts_demo.py```);
7. Rename or delete the TTS folder and download the Assistant and other scripts from this repo 
8. Check everything works *(following)*
9. paste all your keys in the ```.env``` file
<br>

### Checks
- Verify your graphic engine and CUDA version are compatible with PyTorch by running `torch.cuda.is_available()` and `torch.cuda.get_device_name(0)` inside Pyhton; . 
- run ```tests.py```. This file attempt to perform basic operations that might raise errors;
- [WARNING] Check FAQ below if you have errors;
- You can check the sources of error by running demos in the demos folder;


## Step 2: Language support
- To have answers spoken in your language you should first check if your language is supported by the speech generator at __https://cloud.ibm.com/docs/text-to-speech?topic=text-to-speech-voices__; 
- If it's supported, add or change the languages inside ```VirtualAssistant.__init__()``` ;<br>

<p align="center">
  <img src="https://user-images.githubusercontent.com/49094051/230505516-4dba0f29-f45a-4311-aa54-1d93fca25de5.PNG"/>
</p>

- Remember: The loaded Whisper is the medium one. If it performs badly in your language, upgrade to the larger one in the ```__main__()``` at `whisper_model = whisper.load_model("large")`; but I hope your GPU memory is large likewise.

## Step 3: Running (`openai_api_chatbot.py`):
When running, you'll see much information being displayed. I'm constantly striving to improve the readability of the execution, the whole project is a huge beta, forgive slight variations from the screens below. Anyway, this is what happens in general terms when you hit 'run':
- Preliminary initializations take place, you should hear a chime when the Assistant is ready;
- When *awaiting for triggering words* is displayed you'll need to say `ELEPHANT` to summon the assistant. This magic word can be switched, but it needs to be English. At this point, a conversation will begin and you can speak in whatever language you want (if you followed step 2). The conversation will terminate when you say a [stop word](https://github.com/gianmarcoguarnier/JARVIS-ChatGPT/tree/main#key-words) or when you stop making questions for more than 30 seconds (still unstable, needs to be improved) <br>
<p align="center">
  <img src="https://user-images.githubusercontent.com/49094051/230505896-c8a2ff80-4265-41e4-a6d5-e9f56d156afa.PNG" /><br>
  <img src="https://user-images.githubusercontent.com/49094051/230506756-287a1d6b-9652-4c66-bea8-cd75380ab45b.PNG" /><br>
</p>

- After the magic word is said, the word *listening...* should then appear. At this point, you can make your question. When you are done just wait (3 seconds) for the answer to be submitted;
- The script will convert the recorded audio to text using Whisper;
- The text will be analyzed and a decision will be made. If the Assistant believes you want to perform an action (like looking for a past conversation) the respective protocols will be initiated; 
- Elsewise, the script will then expand the `chat_history` with your question it will send a request with the API and it will update the history as soon as it receives a full answer from ChatGPT (this may take up to 5-10 seconds, consider explicitly asking for a short answer if you are in a hurry);
- The `say()` function will perform voice duplication to speak with Jarvis/Someone's voice; if the argument is not in English, IBM Watson will send the response from one of their nice text-to-speech models. If everything fails, the functions will rely on pyttsx3 which is a fast yet not as cool alternative;
<p align="center">

</p>

- When any of the stop keywords are said, the script will ask ChatGPT to give a title to the conversation and will save the chat in a .txt file with the format 'CurrentDate-Title.txt';
- The assistant will then go back to sleep;
<p align="center">
 <img src='https://user-images.githubusercontent.com/49094051/227788180-b9da0957-a58b-4c1c-bc34-4a4c8a0e0957.PNG'/><br>
  <i><span style="color:grey">I made some prompts and closed the conversation</span> </i>
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
- [x] [4  - 2023] Create a full stack ```VirtualAssistant``` class with memory and local storage access
- [x] [4  - 2023] Add sound feedback at different stages (chimes, beeps...)
- [x] [4  - 2023] International language support for voice commands (beta)
- [x] [11  - 2023] Making a step-by-step tutorial 

currently working on:
- [ ] Move some processing locally to reduce credit consumption: [Vicuna: A new, powerful model based on LLaMa, and trained with GPT-4](https://www.youtube.com/watch?v=ByV5w1ES38A&ab_channel=TroubleChute);
- [ ] Integrate with [Auto-GPT](https://github.com/Torantulino/Auto-GPT) which seems to be great to gather material from interned;
- [ ] Improve overall cod;

following:
- [ ] fixing chat length bug (when the chat is too long it can't be processed by ChatGPT 3.5 Turbo)
- [ ] Integrate with Eleven Labs Voices for super expressive voices and outstanding voice cloning;
- [ ] Extending voice commands and *Actions* (make a better active assistant)
- [ ] expanding *Memory*  
- [ ] Connect the system to the internet
- [ ] Refine memory and capabilities
<br>
<br>
### waiting for ChatGPT4 to:
- [ ] add multimodal input (i.e. "Do you think 'this' [holding a paper plane] could fly" -> camera -> ChatGPT4 -> "you should improve the tip of the wings" )
- [ ] Extend *project memory* to images, pdfs, papers...

<span style="color:grey">*Check the [UpdateHistory.md](https://github.com/gianmarcoguarnier/JARVIS-ChatGPT/blob/main/UpdateHistory.md) of the project for more insights.*</span>

Have fun!

# ERRORS and FAQs
categories: Install, General, Runtime
### INSTALL: I have conflicting packages while installing *venv_requirements.txt*, what should I do? <br>
1) Make sure you have the right Python version (3.7) on the venv (>python --version with the virtual environment activated). 2) Try to edit the _venv_requirements.txt_ and remove the version requirements of the incriminated dependencies. 3) Straight remove the package from the txt file and install them manually afterward.<br>

### INSTALL: I meet an error when running openai_api_chatbot.py saying: TypeError: LoadLibrary( ) argument 1 must be str, not None what's wrong? <br>
The problem is concerning Whisper. You should re-install it manually  with ```pip install whisper-openai``` <br>

### INSTALL: I can't import 'openai.embeddings_utils' or 'openai.datalib'<br>
1) Try to ```pip install --upgrade openai```. 
2) This happens because openai elevated their minimum requirements. I had this problem and solved by manually downloading [embeddings_utils.py](https://github.com/openai/openai-python/blob/main/openai/embeddings_utils.py) inside ./<your_venv>/Lib/site-packages/openai/ 
<br>
3) If the problem persists with ```datalib``` raise an issue and I'll provide you the missing file
4) upgrade to python 3.8 (create new env and re-install TTS, requirements)

### INSTALL: I encounter the error ModuleNotFoundError: No module named '\<some module\>' <br>
Requirements are not updated every commit. While this might generate errors you can quickly install the missing modules, at the same time it keeps the environment clean from conflicts when I try new packages (and I try LOTS of them) <br>

### RUN TIME: I encounter some OOM memory when loading the Whisper model, what does it mean?<br>
It means the model you selected is too big for your CUDA device memory. Unfortunately, there is not much you can do about it except load a smaller model. If the smaller model does not satisfy you, you might want to speak 'clearer' or make longer prompts to let the model predict more accurately what you are saying. This sounds inconvenient but, in my case, greatly improved my English-speaking :) <br>

### RUN TIME: Max length tokens for ChatGPT-3.5-Turbo is 4096 but received... tokens.<br>
This is a bug still present, don't expect to have ever long conversations with your assistant as it will simply have enough memory to remember the whole conversation at some point. A fix is in development, it might consist of adopting a 'sliding windows' approach even if it might cause repetition of some concepts. <br>

### GENERAL: I finished my OPENAI credit/demo, what can I do? <br>
1: pay. The price is not that bad and you might end up paying a few dollars a month since pricing depends on usage. 2) create a new account with a new phone number every 3 months 3) Wait for local LLM models to be integrated with the project. 

### GENERAL: For how long will this project be updated? 
Right now (April 2023) I'm working almost non-stop on this. I will likely take a break in the summer because I'll be working on my thesis. 

If you have questions you can contact raise an Issue and I'll do my best to help as soon as possible.
<p align="right"><i>Gianmarco Guarnier<i></p>
