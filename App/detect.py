from keras.models import load_model
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import deque

def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result

#過去7回分のpred履歴(dequeを使用)
pred_history = deque(maxlen=7)
def detect(pred):
  pred_history.append(pred)#pred_historyに追加
  if len(pred_history) > 7:
    pred_history.popleft()#古いpredを削除
  
  #過去5回分のpredの平均を計算
  pred_history_array = np.array(pred_history)
  pred_history_mean = np.mean(pred_history_array, axis=0)
  
  a = ["-","C","G","F","Am","Em","Dm"]
  pred_label = a[np.argmax(pred_history_mean[0])]
  score = np.max(pred_history_mean)
  print(pred_label, score)
  return pred_label, score

def process_audio_stream(stream, chunk, model):
  # ストリームから音声信号を読み取る
  input = stream.read(chunk, exception_on_overflow=False)

  # bufferからndarrayに変換
  ndarray = np.frombuffer(input, dtype='int16')
  
  # FFT変換を行う
  f = np.fft.fft(ndarray)
  
  # その他の処理を行う
  Amp = np.abs(f/(chunk/2))**2
  Amp = min_max(Amp)
  Amp = Amp.reshape(1,32,32,1)
  Amp = Amp.astype('float32')
  
  # モデルを使用して予測を行う
  pred = model.predict(Amp, verbose=0)
  return pred, ndarray

