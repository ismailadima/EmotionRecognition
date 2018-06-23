import numpy as np
from keras.models import load_model
from tkinter import filedialog
from tkinter import *
import re
import librosa
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def openAudio(root):
    root.update()
    root.filename = filedialog.askopenfilename(initialdir="/datapredict", title="Select file",filetypes=(("wav files", "*.wav"), ("all files", "*.*")))
    return root.filename

def avgMFCC(mfcc):
    temp = []
    for i in range(len(mfcc)):
        temp.append(np.average(mfcc[i]))
    return temp

label = ['angry','fear','surprise','happy','sad','disgust','neutral']

# load model
print("[INFO] Load Model...")
model = load_model('model.h5')

print("[ASK] Do you want predict audio? (Y/N)")
answer = input()

while answer:
    if answer == 'Y' or answer == 'y':
        root = Tk()

        audioPath = openAudio(root)
        y, sr = librosa.load(audioPath)

        root.destroy()

        if y is not None:
            drive, path_and_file = os.path.splitdrive(audioPath)

            sub = [m.start() for m in re.finditer('_', audioPath)]
            kelas = audioPath[sub[0] + 1:sub[1]]
            print("nama kelas ",sub, kelas)
            mfcc = librosa.feature.mfcc(y, sr, n_mfcc=12)
            mfccs = avgMFCC(mfcc)

            X = mfccs  # mengambil data pada kolom 0-11
            Y = label.index(kelas)

            # Merubah list ke array numpy (list ke data frame)
            X = np.array(X)
            Y = np.array(Y)

            # Normalized X
            min,max = np.genfromtxt("data_minmax.csv", delimiter=",")
            X = (X - min) / (max - min)

            # Reshape
            X = X.reshape(1, 1, 12, 1)

            # Predict
            probas = model.predict(X)
            classes = probas.argmax()

            # Print prediksi
            print("[PREDICT] Actual Class =", kelas, ", Predict Class =", label[classes])
        else:
            print("[INFO] Audio can't be read")
    elif  answer == 'N' or answer == 'n':
        sys.exit()
    else:
        print("[INFO] Answer must Y/N!")
    print("\n[ASK] Do you want predict audio again? (Y/N)")
    answer = input()