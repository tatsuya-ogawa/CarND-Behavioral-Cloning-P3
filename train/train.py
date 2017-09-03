import csv
import cv2
import os
import numpy as np

lines = []
data_dir = os.path.join('..', 'drive_data')
with open(os.path.join(data_dir, 'driving_log.csv')) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split(os.sep)[-1]
    current_path = os.path.join(data_dir, 'IMG', filename)
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

x_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import Lambda
from keras.layers import Conv2D
from keras.layers import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))

# model.add(Conv2D(20, kernel_size=5, strides=1, activation='relu', input_shape=(160, 320, 3)))
#model.add(MaxPooling2D(2, strides=2))
model.add(Conv2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())

# model.add(Conv2D(50, kernel_size=5, strides=1, activation='relu'))
#model.add(MaxPooling2D(2, strides=2))
model.add(Conv2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

model.save('model.h5')
