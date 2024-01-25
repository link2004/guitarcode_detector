import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from keras.models import load_model
import numpy as np
import pyaudio
import detect
import pygame

CHUNK = 1024
RATE = 44100 #サンプリング周波数
P = pyaudio.PyAudio()
a = ["-","C","G","F","Am","Em","Dm"]

stream = P.open(format=pyaudio.paInt16, channels=1, rate=RATE, frames_per_buffer=CHUNK, input=True, output=False)
model = load_model('model_CNN.h5')

#######Tkinter GUI########

# GUIのメインウィンドウを作成
root = tk.Tk()
root.title("ギターコード検出")

# プロット用のフレームを追加
fig = Figure(figsize=(5, 1), dpi=100)
plot = fig.add_subplot(1, 1, 1)
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# ラベルを追加（検出されたコードを表示）
label_q = tk.Label(root,text="",font=("Helvetica", 100),fg="red")
label_q.pack()
label_ans = tk.Label(root, text="",font=("Helvetica", 100))
label_ans.pack()

######mp3再生########
def play_mp3(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    
######question_label########

question_label = None
def shuffle_question_label():
  global question_label
  question_label = a[np.random.randint(2,7)]
  #音を鳴らす
  #play_mp3(f"mp3/{question_label}.mp3")
  return question_label
shuffle_question_label()

#######stream_process()########

# ストリーム処理関数をGUIに組み込む
def stream_process():
    # 省略：ストリームからのデータ読み取りと処理
    pred, ndarray = detect.process_audio_stream(stream, CHUNK, model)
    pred_label, score =detect.detect(pred)
  
    plot.clear()
    plot.plot(ndarray, color='blue')
    canvas.draw()

    # ここでpred_labelとscoreを取得し、ラベルに表示
    # label.config(text=f"検出されたコード: {pred_label}, スコア: {score:.2f}")
    label_q.config(text= question_label)
    label_ans.config(text= pred_label)
    if(pred_label == question_label):
      shuffle_question_label()
      while(pred_label == question_label):
        shuffle_question_label()
    elif(pred_label != "-"):
      #play_mp3(f"mp3/{question_label}.mp3")
      pass
    # 関数を繰り返し実行
    root.after(10, stream_process)








# ストリーム処理関数を初回実行
root.after(10, stream_process)

# メインループ
root.mainloop()

# ストリームの停止と終了処理
stream.stop_stream()
stream.close()
P.terminate()
