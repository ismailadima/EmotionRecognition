import numpy as np
import librosa

label = ['angry','fear','surprise','happy','sad','disgust','neutral']

def avgMFCC(mfcc,kelas):
    temp = []
    for i in range(len(mfcc)):
        temp.append(np.average(mfcc[i]))
    temp.append(kelas+1)
    return temp

dataMFCC = []
dataLabel = []
for j in range(1,201):
    for i in range(len(label)):
        y, sr = librosa.load("dataset/OLDER_"+label[i]+"/older_"+label[i]+"_ ("+str(j)+").wav")
        mfcc = librosa.feature.mfcc(y,sr,n_mfcc=12)
        mfccs = avgMFCC(mfcc,i)
        dataMFCC.append(mfccs)

        y, sr = librosa.load("dataset/YOUNG_"+label[i]+"/young_"+label[i]+"_ ("+str(j)+").wav")
        mfcc = librosa.feature.mfcc(y,sr,n_mfcc=12)
        mfccs = avgMFCC(mfcc,i)
        dataMFCC.append(mfccs)


print(len(dataMFCC), dataMFCC[0])

np.savetxt("data_mfcc.csv", dataMFCC, fmt="%.8f", delimiter=",")
