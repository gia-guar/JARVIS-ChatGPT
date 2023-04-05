import whisper
import pyaudio

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
SILENCE_THRESHOLD = 1500

# convert audio content into text
def whisper_wav_to_text(audio_name, model=[], model_name=False, prior = None):
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
        if not(prior is None):
            filt_probs = {str(lan):probs.get(lan) for lan in prior}
            probs = filt_probs
        print(f"Detected language: {max(probs, key=probs.get)}")
        detected_lang = str(max(probs, key=probs.get))

        options = whisper.DecodingOptions(language=detected_lang)
    except:
        # model does not support multiple languages, default to English
        print('language: en')
        detected_lang = 'en'
        options = whisper.DecodingOptions(language='en')
    
    result = whisper.decode(model, mel, options)

    # print the recognized text
    print('\n[User]: '+ result.text)
    return result.text, detected_lang
