import os
import wave

import pylab
def graph_spectrogram(wav_file, name, dir):
    sound_info, frame_rate = get_wav_info(wav_file)
    pylab.figure(num=None, figsize=(19, 12))
    pylab.subplot(111)
    pylab.specgram(sound_info, Fs=frame_rate)
    pylab.axis(False)
    pylab.savefig(dir + name[:len(name) - 4] + '.png')

def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.fromstring(frames, 'int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate

names = os.listdir('Testing_Data')
dir = 'Testing_Data/'
upload_dir = 'spectrogram_train/test/'

for name in names:
    graph_spectrogram(dir + name, name, upload_dir)