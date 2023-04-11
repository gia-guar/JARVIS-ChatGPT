import Assistant.get_audio as myaudio

print('get_devices() print all devices:')
print(myaudio.get_devices())

print('\n\n')

print('detect_microphones() print mics only:')
print(myaudio.detect_microphones())

print('\n')
print('get_device_channels() return channel count for each device\n', myaudio.get_device_channels())

# select the first available microphone
M = myaudio.get_devices()[0]['name']
print(M)