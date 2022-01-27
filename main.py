import os

import matplotlib.pyplot as plt
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
import librosa
import numpy as np

def features_extractor(name):
    audio, sample_rate = librosa.load(name, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)

    return mfccs_scaled_features

X_test = []
X = []
Y = []

names = os.listdir('training_data/human')
dir = 'training_data/human/'

for name in names:
    X.append(features_extractor(dir + name))
    Y.append(1)

names = os.listdir('training_data/spoof')
dir = 'training_data/spoof/'

for name in names[:10322]:
    X.append(features_extractor(dir + name))
    Y.append(0)

names_test = os.listdir('Testing_Data')
dir = 'Testing_Data/'

for name in names_test:
    X_test.append(features_extractor(dir + name))

X = np.array(X)
X_test = np.array(X_test)
Y = utils.to_categorical(Y, 2)

print(X.shape, Y.shape, X_test.shape)

myAdam = Adam(learning_rate = 0.01)
myOpt = SGD(learning_rate = 0.02, momentum = 0.3, nesterov = True)

model=Sequential([
    Dense(128, input_shape=(40, ), activation = 'relu'),
    BatchNormalization(),
    # Dropout(0.3),
    Dense(64, activation = 'relu'),
    BatchNormalization(),
    # Dropout(0.3),
    # Dense(32, activation = 'relu'),
    # BatchNormalization(),
    # Dropout(0.3),
    Dense(2, activation = 'softmax')
])

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = 'accuracy')

history = model.fit(X, Y, batch_size = 32, epochs = 50, shuffle = True, validation_split = 0.2)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.grid(True)
plt.show()

Y_test = model.predict(X_test)

f = open('spectrogram_train/sample.txt', 'w')

for i in range(len(names_test)):
    f.writelines(names_test[i][:len(names_test[i]) - 4] + '.wav, ' + str(Y_test[i][1]) + '\n')

f.close()
