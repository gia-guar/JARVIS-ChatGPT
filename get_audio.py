import whisper
import wave
import pyaudio
import speech_recognition as sr
import audioop
import math
import time

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
SILENCE_THRESHOLD = 1500

# convert audio content into text
def whisper_wav_to_text(audio_name, model=[], model_name=False):
    if isinstance(model_name, str):
        print('loading model ', model_name)
        model = whisper.load_model(model_name)

    if model == []:
        raise Exception("model cannot be unspecified")

    print('listening to ',audio_name,'...')
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio_name)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    try:
        _, probs = model.detect_language(mel)
        print(f"Detected language: {max(probs, key=probs.get)}")
        detected_lang = str(max(probs, key=probs.get))
        options = whisper.DecodingOptions()
    except:
        # model does not support multiple languages, default to English
        print('language: en')
        detected_lang = 'en'
        options = whisper.DecodingOptions(language='en')
    
    result = whisper.decode(model, mel,options)

    # print the recognized text
    print('\n[User]: '+ result.text)
    return result.text, detected_lang






def PassiveListening(KeyWords=["elephant"], verbosity=False):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        if verbosity: print("listenting passively...", end="")
        audio = r.listen(source)
        query = ''

        try: 
            query = r.recognize(audio)
            if verbosity: print(f"user said: {query}")
        except Exception as e:
            if verbosity: print(str(e))
    
    if any(item in query.split() for item in KeyWords): return True
    else: return False





def record(KeyWords=["elephant"], SleepTimer=-10000, verbosity=False): 
    RESPONSE_TIME = 3
    SLEEP_DELAY = 30

    # if there isn't any conversation ongoing (last ten minutes) wait for the triggering word
    now = time.perf_counter()
    if now - SleepTimer > 600:
        isAwake = False
        print('[assistant went asleep]')
        print('waiting for triggering words')
    else: isAwake = True

    ## SUMMONING:
    while not(isAwake):
        isAwake = PassiveListening(KeyWords=KeyWords, verbosity=verbosity)


    ## init Microphone streamline
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    frames = []
    try:
        silence_time = 0
        speaked = False

        print("listening...")
        while True:
            delta = time.perf_counter()
            data = stream.read(CHUNK)
            frames.append(data)
                  
            # detect silence
            data = stream.read(CHUNK)
            sound_amplitude = audioop.rms(data, 2)
            delta = time.perf_counter() - delta

            if(sound_amplitude < SILENCE_THRESHOLD):      
                silence_time = silence_time + delta

                # break the loop and return the audio
                if silence_time > RESPONSE_TIME and speaked:
                    raise KeyboardInterrupt   
                
                if silence_time>SLEEP_DELAY:
                    break
            else:
                speaked = True
                silence_time = 0

    except KeyboardInterrupt:
        print("Done recording")
    except Exception as e:
        print(str(e))

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    return sample_width, frames

def record_to_file(file_path, SleepTimer = -10000):
	wf = wave.open(file_path, 'wb')
	wf.setnchannels(CHANNELS)
	sample_width, frames = record(SleepTimer=SleepTimer)
	wf.setsampwidth(sample_width)
	wf.setframerate(RATE)
	wf.writeframes(b''.join(frames))
	wf.close()


def demo():
    model = whisper.load_model("base.en")
    print('#' * 80)
    print("Please speak words into the microphone")
    print('Press Ctrl+C to stop the recording')
    
    # record
    record_to_file('audio.wav')

    print("Result written to output.wav")
    print("\n## transcribing ##")

    # transcribe
    text = whisper_wav_to_text('output.wav',model)
    print('#' * 80)