#импортируем библиотеки
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import img_to_array
from skimage.color import rgba2rgb
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

imgs = []
Y = []

names = os.listdir('spectrogram_train/human')
dir = 'spectrogram_train/human/'

for name in names[:500]:
    imgs.append(dir + name)
    Y.append(1)

names = os.listdir('spectrogram_train/spoof')
dir = 'spectrogram_train/spoof/'

for name in names[:500]:
    imgs.append(dir + name)
    Y.append(0)

Y = to_categorical(Y, 2)

X = []
for img in imgs:
    X.append(img_to_array(rgba2rgb(Image.open(img))))


X = np.array(X)
print(X.shape)

model = Sequential([
    Conv2D(64, (3, 3), padding = 'same', activation = 'relu', batch_input_shape = (100, 1200, 1900, 3)),
    MaxPooling2D((2, 2), strides = 2),
    Conv2D(128, (3,3), padding = 'same', activation = 'relu'),
    MaxPooling2D((2, 2), strides = 2),
    Conv2D(256, (3, 3), padding='same', activation='relu'),
    MaxPooling2D((2, 2), strides=2),
    Conv2D(512, (3, 3), padding='same', activation='relu'),
    MaxPooling2D((2, 2), strides=2),
    Flatten(),
    Dense(512, activation = 'relu'),
    Dense(2, activation = 'softmax')
])

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])
history = model.fit(X, Y, batch_size=100, shuffle=True, epochs = 10, validation_split=0.2)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.grid(True)
plt.show()

imgs_test = []

names_test = os.listdir('spectrogram_train/test')
dir = 'spectrogram_train/test/'

for name in names_test:
    imgs_test.append(dir + name)

X_test = []

for img in imgs_test:
    X_test.append(img_to_array(rgba2rgb(Image.open(img))))

Y_test = model.predict(X_test)

f = open('spectrogram_train/sample.txt', 'w')

for name in names_test:
    f.write(name[:len(name) - 4] + '.wav' + str(Y[:, 1]))

f.close()