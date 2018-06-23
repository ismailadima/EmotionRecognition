import keras
import numpy as np
from keras.models import Sequential
from keras.layers import *
from sklearn.model_selection import KFold

# Setting Network dan Training
NB_EPOCH = 25
VERBOSE = 0 #Verbosity mode 0 = silent, 1 = progress bar, 2 = one line per epoch.
NB_CLASSES = 7  #jumlah output emosi (kelas target)
N_HIDDEN = 1000 #jumlah hidden layer
N_LINES = 1   #jumlah data
N_INPUTFILES = 2800
BATCH_SIZE = 10

# LOAD MFCC
data = np.genfromtxt("data_mfcc.csv", delimiter=",")
X = [i for i in data[:,0:12]]
Y = [int(i-1) for i in data[:,-1]]

X = np.array(X)
Y = np.array(Y)

# Normalized X
max = np.amax(X)
min = np.amin(X)
X_normalized = (X-min) / (max-min)

# Convert class vector ke binnary class matrics
# label pada Y = 0: 'angry', 1: 'fear',2: 'surprise',3: 'happy',4: 'sad',5: 'disgust',6: 'neutral'
Y = keras.utils.to_categorical(Y, NB_CLASSES)

# Reshape ke [samples][pixels][width][height]
X = X.reshape(N_INPUTFILES, N_LINES, 12, 1)
X_normalized = X_normalized.reshape(N_INPUTFILES, N_LINES, 12, 1)

# Membuat model CNN
model = Sequential()
model.add(Conv2D(20, kernel_size = 5, padding = "same", input_shape = (N_LINES, 12, 1)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides= (2,2), data_format="channels_first"))
model.add(Flatten())
model.add(Dense(N_HIDDEN))
model.add(Activation("relu"))

# A softmax classifier
model.add(Dense(NB_CLASSES))
model.add(Activation("softmax"))
model.summary()

# compile model dengan Adam optimizer
model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model_normalize = model

# fold validation k = 10
k_fold = KFold(n_splits=10)
k = 0
sum_score = 0
sum_score_normalized = 0
max_score = 0
max_score_normilazed = 0
best_indices = () #train, test
best_indices_normalized = ()

for train_indices, test_indices in k_fold.split(X):
    # penambahan k untuk print
    k += 1

    # pengambilan data uji dan data latih
    X_train = X[train_indices]
    X_train_normalized = X_normalized[train_indices]
    Y_train = Y[train_indices]

    X_test = X[test_indices]
    X_test_normalized = X_normalized[test_indices]
    Y_test = Y[test_indices]

    # Fit model
    model_normalize.fit(X_train_normalized, Y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE)
    model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE)

    # evaluasi
    score = model.evaluate(X_test, Y_test)
    score_normilized = model_normalize.evaluate(X_test_normalized, Y_test)

    if max_score < score[1]:
        max_score = score[1]
        best_indices = (train_indices, test_indices)

    if max_score_normilazed < score_normilized[1]:
        max_score_normilazed = score_normilized[1]
        best_indices_normalized = (train_indices, test_indices)

    print("\n###########################################################################  Fold ke-{}\n".format(k))
    print("Tanpa Normalisasi\n{}: {}".format(model.metrics_names[1], score[1] * 100))
    print("Dengan Normalisasi\n{}: {}".format(model_normalize.metrics_names[1], score_normilized[1] * 100))
    print("\n###########################################################################\n")

    sum_score += score[1] * 100
    sum_score_normalized += score_normilized[1] * 100

avg_score = sum_score/10
avg_score_normalized = sum_score_normalized/10

print("\nRata-rata Akurasi")
print("Tanpa Normalisasi : {}".format(avg_score))
print("Dengan Normalisasi : {}".format(avg_score_normalized))


print("\nAkurasi Terbaik")
if max_score > max_score_normilazed :
    print("Tanpa normalisasi, dengan score {} dan iterasi (train, test) = {}".format(max_score), best_indices)
elif max_score < max_score_normilazed :
    print("Dengan normalisasi, dengan score {} dan iterasi (train, test) = {}".format(max_score_normilazed), best_indices_normalized)
else:
    print("Tanpa maupun dengan normalisasi memiliki score yang sama yait {}, dengan iterasi (train, test) = {}".format(max_score), best_indices)