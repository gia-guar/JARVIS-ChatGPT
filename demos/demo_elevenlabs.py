import elevenlabslib
from dotenv import load_dotenv
import os
from pydub import AudioSegment
import io
import pygame

load_dotenv()

api_key = os.getenv('ELEVENLABS_API_KEY')

user = elevenlabslib.ElevenLabsUser(api_key)

text = "This is the documentation for the ElevenLabs API. You can use this API to use our service programmatically, this is done by using your xi-api-key"
write_dir = os.path.join('Assistant', 'answers')

elevenlabs_voice = user.get_voices_by_name('Antoni')[0]
audio = elevenlabs_voice.generate_audio_bytes(text)

audio = AudioSegment.from_file(io.BytesIO(audio), format="mp3")
audio.export(os.path.join(write_dir, "speech.wav"), format="wav")
if pygame.mixer.get_init() is None: pygame.mixer.init()
try:
    pygame.mixer.music.load(os.path.join(write_dir, "speech.wav"))
except Exception as e:
    print(e)

pygame.mixer.music.set_volume(0.5)
pygame.mixer.music.play()
while(pygame.mixer.music.get_busy()):pass