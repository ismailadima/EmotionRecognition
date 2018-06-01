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
for i in range(len(label)):
    for j in range(0,199):
        y, sr = librosa.load("Dataset/OLDER_"+label[i]+"/older_"+label[i]+"_ ("+str(j+1)+").wav")
        mfcc = librosa.feature.mfcc(y,sr,n_mfcc=12)
        mfccs = avgMFCC(mfcc,i)
        dataMFCC.append(mfccs)

np.savetxt("data_mfcc.txt", dataMFCC, fmt="%.8f", delimiter=",")