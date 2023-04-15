from TTS.api import TTS
import os


"""
DESCRIPTION: Clone some voices in Real Time with TTS! run the scripts and listen to output.wav
"""

TEXT = """ OpenAI provides a toolkit called "OpenAI API" that allows you to compute multi-head self-attention for a given text. The OpenAy I A Pi I provides access to a range of pre-trained language models, such as Gee Pee Tee 2 and Gee Pee Tee 3, which are capable of performing multi-head self-attention on text data.
To use the OpenAI API, you first need to sign up for an API key and then import the A P I client into your Python environment. Once you have set up the A P I client, you can use it to send requests to the OpenAI API server, which will process your text data and return a response containing the multi-head self-attention weights.
The OpenAI API offers a simple and straightforward way to access powerful language models and compute complex natural language processing tasks, such as multi-head self-attention, without the need for extensive computational resources or expertise in machine learning.
"Exploring Multi-Head Self-Attention for Keyword Identification in Natural Language Processing"

"""

#   Running a multi-speaker and multi-lingual model

#   List available 🐸TTS models and choose the first one
model_name = TTS.list_models()[0]
#   Init TTS
tts = TTS(model_name)
#   Run TTS
#    ❗ Since this model is multi-speaker and multi-lingual, we must set the target speaker and the language
#   Text to speech with a numpy output

wav = tts.tts("This is a test! This is also a test!!", speaker=tts.speakers[0], language=tts.languages[0])
#    Text to speech to a file
tts.tts_to_file(text="Hello world!", speaker=tts.speakers[0], language=tts.languages[0], file_path="output.wav")

#   Running a single speaker model

#   Init TTS with the target model name
tts = TTS(model_name="tts_models/de/thorsten/tacotron2-DDC", progress_bar=False, gpu=False)

#   Run TTS
tts.tts_to_file(text="Ich bin eine Testnachricht.", file_path='.')

#   Example voice cloning with YourTTS in English, French and Portuguese:
tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=True)
tts.tts_to_file(text=TEXT, speaker_wav=os.path.join(os.getcwd(),'voices','JARVIS','PaulBettanyLongMP3.mp3'), language="en", file_path="output.wav")
