# imports
import pyttsx3
from ibm_watson import TextToSpeechV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from TTS.api import TTS
import os


class Voice:
    def __init__(self, **kwargs):   
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

        except:
            print('IBM authentication failed')

        # PYTTSX3
        engine = pyttsx3.init()

        # SYNTHETIC VOICES

        # # CorentinJ     - Real Time Voice Cloning github (https://github.com/CorentinJ/Real-Time-Voice-Cloning)
        #[deprecated] synth = JARVIS.init_jarvis()   

        # CoquiAI         - coqui-ai/TTS (https://github.com/coqui-ai/tts)
        synth = TTS(model_name=os.path.join("tts_models/multilingual/multi-dataset/your_tts"), progress_bar=False, gpu=True)  

        self.tts_service = [tts]
        self.path = kwargs['voice_id']
        self.synthetic_voice = synth
        self.offline = engine
    

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
