import numpy as np
import librosa

label = ['angry','fear','surprise','happy','sad','disgust','neutral']

def avgMFCC(mfcc,kelas):
    temp = []
    for i in range(len(mfcc)):
        temp.append(np.average(mfcc[i]))
    temp.append(kelas)
    return temp

dataMFCC = []
dataLabel = []
for j in range(1,200):
    for i in range(len(label)):
        y, sr = librosa.load("dataset/OLDER_"+label[i]+"/older_"+label[i]+"_ ("+str(j)+").wav")
        mfcc = librosa.feature.mfcc(y,sr,n_mfcc=12)
        mfccs = avgMFCC(mfcc,i)
        dataMFCC.append(mfccs)

        y, sr = librosa.load("dataset/YOUNG_"+label[i]+"/young_"+label[i]+"_ ("+str(j)+").wav")
        mfcc = librosa.feature.mfcc(y,sr,n_mfcc=12)
        mfccs = avgMFCC(mfcc,i)
        dataMFCC.append(mfccs)

np.savetxt("data_mfcc.csv", dataMFCC, fmt="%.8f", delimiter=",")
print("Data MFCC Tersimpan")

