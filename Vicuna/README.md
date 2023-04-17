# OOBABOOGA-UI & VICUNA INSTALLATION
Guide on how to run a ChatGPT alternative on your PC.

## 1. What are the OOBABOOGA-UI and Vicuna?
The [oobabooga text generation web ui](https://github.com/oobabooga/text-generation-webui) is a User Interface that can utilize many open-source Large Language Models like Alpaca, Llama and many others. **It's 100% free and runs on your pc so no connection is required**. The powerful aspect of the Oobabooga interface is that it provides a way to interface with the model through an Application Program Interface, which is handy for this project. <br>
<p>
    <img width="2348" alt="Vicuna" src="https://raw.githubusercontent.com/oobabooga/screenshots/main/cai3.png" width="10">
</p>
<p align='center'>
    <span style="color:grey"> Example of conversation in Oobabooga UI </span>
</p>

[Vicuna 13-b](https://vicuna.lmsys.org/) is an open-source chatbot trained by fine-tuning LLaMA on user-shared conversations collected from ShareGPT. Preliminary evaluation using GPT-4 as a judge shows Vicuna achieves more than 90% quality of OpenAI ChatGPT. 

## 2. Should I install this? (*)
Processing large amounts of information can be quite expensive if you are running on a pay-as-you-go account. Secondly, having a backup solution for when our connection isn't reliable or just when there is too much traffic on OpenAI servers has its pros and cons. For this project, I am planning on using offline, free text processing for PDFs and long website analysis. In this way It'll be possible to let the computer elaborate huge amounts of material in the background while I'll be doing other stuff.

## 3. (*) Pros and cons
Pros:
 1. **local**, no internet needed;
 2. **free**;
 3. reasonably good *(see cons: hallucinations);
 4. runs on CPU and/or GPU;
 5. You have some freedom in choosing what opensource model works best for you;

Cons:
 1. **Size**: these are **LARGE** language models. They are designed to take as much RAM/vRAM on your devices. This means that, depending on the performance of your hardware, you might face Out Of Memory errors.
 <p align="center">
    <img width="2348" alt="Vicuna" src="https://user-images.githubusercontent.com/49094051/232077149-882178eb-c73e-4834-b82e-44cafa941666.PNG">
</p>

> This snap was taken running Whisper's "large" model (10GB) and the GPU Vicuna model. In particular, you can see the RAM getting filled when the model begins to process.

 2. **Speed**: OpenAI services are generally faster. The speed of the answer will depend on your hardware;
 3. **Multitasking**: due to their size and RAM footprint, multitasking with these software running might be difficult;
 4. Hallucinations: these models are quite raw and they tend to have long discussions... with themselves! Coding can put a patch on the problem but right now it's still an issue and, sometimes, answers might be *weird*. <span style="color:grey"> For example, Vicuna might ask itself a question after it gave an answer and go on like that until it consumes all tokens!</span>

# INSTALLATION
## 1. Make a folder anywhere on your computer and click on the folder path:
 <p align="center">
    <img width="2348" alt="Vicuna" src="https://user-images.githubusercontent.com/49094051/232081647-5c5ccc3e-1fc0-45d8-905b-4c91ac67e77f.png">
    <span style='color:grey'> Here <i>ChatGPT</i> is where I am keeping the project, while <i>vicuna</i> is the folder I chose to install Vicuna in </span>
</p>


## 3. Type ```powershell``` and hit enter

## 4. run ```iex (irm vicuna.tc.ht)``` from powershell
This command will run a One-Click-Installation provided by [TroubleChute](https://hub.tcno.co/ai/text-ai/vicuna/). You can follow the instructions you'll see on screen or check his [step-by-step tutorial](https://youtu.be/d4dk_7FptXk). 

## 5. Select CPU or GPU (or both)
The Vicuna CPU model ([eachadea/ggml-vicuna-13b-4bit](https://huggingface.co/eachadea/legacy-ggml-vicuna-13b-4bit)) weighs ~15 GB. It can run only on CPU (RAM);<br>
The Vicuna GPU model ([anon8231489123/vicuna-13b-GPTQ-4bit-128g](https://huggingface.co/anon8231489123/vicuna-13b-GPTQ-4bit-128g)) weighs ~7.5 GB. It can dynamically allocate itself between RAM and vRAM. <br>

You can also download manually using the following syntax on CMD: <br>
```cd oobabooga-windows\text-generation-webui```<br>
```python download-model.py facebook/opt-1.3b```<br>
[Hugging Face](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads) is the main place to download models. These are some examples:
* [Pythia](https://huggingface.co/models?sort=downloads&search=eleutherai%2Fpythia+deduped)
* [OPT](https://huggingface.co/models?search=facebook/opt)
* [GALACTICA](https://huggingface.co/models?search=facebook/galactica)
* [GPT-J 6B](https://huggingface.co/EleutherAI/gpt-j-6B/tree/main)


## 6. Edit the ```start-webui-vicuna.bat``` or run ```start-webui-vicuna-gpu.bat```
change the last line with:<br>
 - if you are using GPU:
``` call python server.py --wbits 4 --groupsize 128 --listen --no-stream --model anon8231489123_vicuna-13b-GPTQ-4bit-128g --notebook --extension api```
 - If you are using CPU: ```call python server.py --model eachadea_ggml-vicuna-13b-4bit --listen --no-stream --notebook --extension api```
<br>
This edit is already applied in the file ```start-webui-vicuna-gpu.bat``` you'll find in this folder
You should be able to run the ```.bat``` file with no errors and the following message should appear: <br>Running on local URL:  http://0.0.0.0:786<br>
To create a public link, set `share=True` in `launch()`.
