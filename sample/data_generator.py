import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import math
import csv
CHUNK = 1024
RATE = 44100 #サンプリング周波数
P = pyaudio.PyAudio()
LABEL = "noise"# ここに好きなラベルを設定

a = {v: i for i, v in enumerate(["noise","C","G","F","Am","Em","Dm"])}
label_index = a[LABEL]

stream = P.open(format=pyaudio.paInt16, channels=1, rate=RATE, frames_per_buffer=CHUNK, input=True, output=False)
x = np.arange(1,1025,1)
freq = np.linspace(0, RATE, CHUNK)

#正規化
def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result

file_name = LABEL + ".csv"
path = "./csv/" + file_name
o = open(path,'a') #コードごとにファイルを変更する
writer = csv.writer(o, lineterminator=',\n')

# データの数を追跡するカウンタを追加
data_count = 0

while stream.is_active():
    try:
        input = stream.read(CHUNK, exception_on_overflow=False)
        # bufferからndarrayに変換
        ndarray = np.frombuffer(input, dtype='int16')

        #フーリエ変換
        f = np.fft.fft(ndarray)

        #周波数
        freq = np.fft.fftfreq(CHUNK, d=44100/CHUNK)
        Amp = np.abs(f/(CHUNK/2))**2
        Amp = min_max(Amp)

        # ラベルをデータに追加
        data_with_label = np.append(Amp, label_index)
        # NumPy配列をリストに変換
        data_with_label = data_with_label.tolist()
        writer.writerow(data_with_label)
        print(Amp)

        #フーリエ変換後のスペクトルを表示
        line, = plt.plot(freq[1:int(CHUNK/2)], Amp[1:int(CHUNK/2)], color='blue')
        plt.pause(0.01)
        plt.ylim(0,1)
        ax = plt.gca()
        ax.set_xscale('log')
        line.remove()


        # データの数を増やす
        data_count += 1
        # データの数が1024を超えたらループを抜ける
        if data_count >= 1024:
            break
    except KeyboardInterrupt:
        break

stream.stop_stream()
stream.close()
P.terminate()
f.close()

print('Stop Streaming')
