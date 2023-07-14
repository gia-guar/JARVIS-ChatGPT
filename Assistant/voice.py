# imports
import pyttsx3
from ibm_watson import TextToSpeechV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from TTS.api import TTS
import os
import elevenlabslib
from contextlib import contextmanager
import pygame
from pydub import AudioSegment
import io
import sys
import langid

class Voice:
    def __init__(self, languages, **kwargs):   
        # IBM CLOUD
        try:
            print('Authorizing IBM Cloud:')
            url = kwargs['ibm_url']
            apikey = kwargs['ibm_api']
            # Setup Service
            print('  1/3: Setting up cloud authenticator...')
            authenticator = IAMAuthenticator(apikey)
            # New tts service
            print('  2/3: Setting up text-to-speech...')
            tts = TextToSpeechV1(authenticator=authenticator)
            # set serive url
            print('  3/3: Setting up cloud service ...')
            tts.set_service_url(url)
            print('    ✓ service established\n')
            self.tts_service = tts
        except:
            print('IBM authentication failed')

        if 'elevenlabs_api' in kwargs:
            try:
                eleven_labs_user = elevenlabslib.ElevenLabsUser(kwargs['elevenlabs_api'])
                
                if 'elevenlabs_voice' in list(kwargs.keys()):
                    if kwargs['elevenlabs_voice'] in (voice.initialName for voice in eleven_labs_user.get_all_voices()):             
                        self.elevenlabs_voice = eleven_labs_user.get_voices_by_name(kwargs['elevenlabs_voice'])[0]

            except:
                print('Couldn t connect with Elevenlabs')
            # <to do: initiate Jarvis cloned voice if available and disable TTS>

        # PYTTSX3 for backup plan
        engine = pyttsx3.init()

        # SYNTHETIC VOICES
        # CoquiAI -  coqui-ai/TTS (https://github.com/coqui-ai/tts)
        synth = TTS(model_name=os.path.join("tts_models/multilingual/multi-dataset/your_tts"), progress_bar=False, gpu=True)  
        
        self.languages = languages
        self.write_dir = kwargs['write_dir']
        self.path = kwargs['voice_id']
        print('cloning voice form:',self.path)
        self.synthetic_voice = synth
        self.offline = engine




    def speak(self, text, VoiceIdx, mode, elevenlabs=False, IBM=False):
        ## delete old last_aswer.wav to avoid conflicts
        if os.path.exists((self.write_dir, "last_answer.wav")): os.remove((self.write_dir, "last_answer.wav"))

        ## generate the speech: last_answer.wav
        if mode == 'online':
            if  elevenlabs==True:
                if  VoiceIdx == 'en':
                    try:
                        audio = self.elevenlabs_voice.generate_audio_bytes(text)
                        audio = AudioSegment.from_file(io.BytesIO(audio), format="mp3")
                        audio.export(os.path.join(self.write_dir, "last_answer.wav"), format="wav")
                    except Exception as e:
                        print(f'Elevenlabs credit might have ended. {e}')
                        raise Exception

                if VoiceIdx == 'jarvis':
                    # to do: use voice duplication from elevenlabs
                    print('(ElevenLabs Jarvis voice not yet available)')
                    raise Exception()

            elif IBM==True:
                with open(os.path.join(self.write_dir, "last_answer.wav"),'wb') as audio_file:
                    try:
                        if VoiceIdx=='jarvis':VoiceIdx='en'
                        res = self.tts_service.synthesize(text, accept='audio/wav', voice=get_ibm_voice_id(VoiceIdx)).get_result()
                        audio_file.write(res.content)
                    except:
                        print('(IBM credit might have ended)')
                        raise Exception

        if mode == 'offline': 
            if VoiceIdx == 'jarvis' and langid.classify(text)[0]=='en':
                LangIdx = 'en'
                print(self.path, LangIdx)
                self.synthetic_voice.tts_to_file(text=text, speaker_wav=self.path[LangIdx], language=LangIdx, file_path=os.path.join(self.write_dir, 'last_answer.wav'))
                
                
                """ Idea for multiple language Text-To-Speech: dictionaries
                if VoiceIdx == 'other-language':
                    self.synthetic_voice['other-language'].tts_to_file(text=text, speaker_wav=self.path, language="en", file_path=os.path.join(self.DIRECTORIES['SOUND_DIR'], 'last_answer.wav'))
                """
            else:
                LangIdx = langid.classify(text)[0]
                self.offline = self.change_offline_lang(lang_id=LangIdx)
                self.offline.say(text)
                self.offline.runAndWait()
                return
        
        # play the generated speech:
        if pygame.mixer.get_init() is None:pygame.mixer.init()
        pygame.mixer.music.load(os.path.join(self.write_dir, 'last_answer.wav'))
        pygame.mixer.music.set_volume(0.5)
        pygame.mixer.music.play()
        while(pygame.mixer.music.get_busy()): pass
        return



    def change_offline_lang(self, lang_id):
        engine = pyttsx3.init()
        try:
            for voice in self.offline.getProperty('voices'):
                if self.languages[lang_id] in voice.name:
                    engine.setProperty('voice', voice.id)
                    return engine
            return engine
        except Exception as e:    
            print('error while switching to lang: ',lang_id,e)
            return engine 

# know more at: https://cloud.ibm.com/docs/text-to-speech?topic=text-to-speech-voices
def get_ibm_voice_id(VoiceIdx):
    voices={
        'ar':'ar-MS_OmarVoice',

        'zh':'zh-CN_LiNaVoice',
        'zh':'zh-CN_WangWeiVoice',
        'zh':'zh-CN_ZhangJingVoice',

        'cz':'cs-CZ_AlenaVoice',

        'nl':'nl-BE_AdeleVoice',
        'nl':'nl-BE_BramVoice',
        'nl':'nl-NL_EmmaVoice',
        'nl':'nl-NL_LiamVoice',
        'nl':'nl-NL_MerelV3Voice',

        'en':'en-GB_CharlotteV3Voice',
        'en':'en-GB_JamesV3Voice',
        'en':'en-GB_KateV3Voice',
        'en':'en-US_AllisonV3Voice',
        'en':'en-US_EmilyV3Voice',
        'en':'en-US_HenryV3Voice',
        'en':'en-US_KevinV3Voice',
        'en':'en-US_LisaV3Voice',
        'en':'en-US_MichaelV3Voice',
        'en':'en-US_OliviaV3Voice',

        'fr':'fr-CA_LouiseV3Voice',
        'fr':'fr-FR_NicolasV3Voice',
        'fr':'fr-FR_ReneeV3Voice',

        'de':'de-DE_BirgitV3Voice',
        'de':'de-DE_DieterV3Voice',
        'de':'de-DE_ErikaV3Voice',

        'it':'it-IT_FrancescaV3Voice',
        'ja':'ja-JP_EmiV3Voice',
        
        'ko':'ko-KR_HyunjunVoice',
        'ko':'ko-KR_SiWooVoice',
        'ko':'ko-KR_YoungmiVoice',
        'ko':'ko-KR_YunaVoice',
        'ko':'ko-KR_JinV3Voice',

        'pt':'pt-BR_IsabelaV3Voice',

        'es':'es-ES_EnriqueV3Voice',
        'es':'es-ES_LauraV3Voice',
        'es':'es-LA_SofiaV3Voice',
        'es':'es-US_SofiaV3Voice',

        'sv':'sv-SE_IngridVoice'
        }
    return voices[VoiceIdx]


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout