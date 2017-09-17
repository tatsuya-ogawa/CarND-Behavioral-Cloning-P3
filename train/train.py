import csv
from scipy import signal

import cv2
import os
import numpy as np


class TrainData:
    def __init__(self, y_label, path, direction):
        self.y_label = y_label
        self.path = path
        self.direction = direction


def get_train_generator(sample_count, batch_size, bins_num, test_size):
    lines = []
    data_dir = os.path.join('..', 'drive_data')
    with open(os.path.join(data_dir, 'driving_log.csv')) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    train_sets = []
    # images = []
    measurements = []
    correction = 0.1
    for line in lines:
        for i in range(3):
            source_path = line[i]
            filename = source_path.split(os.sep)[-1]
            current_path = os.path.join(data_dir, 'IMG', filename)
            measurement = float(line[3]) - correction + i * correction
            measurements.append(measurement)
            measurements.append(-measurement)
            train_sets.append(TrainData(measurement, current_path, 1))
            train_sets.append(TrainData(-measurement, current_path, -1))

    y_train = np.array(measurements)

    def resample_data(y_train, bins_num):
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1)
        # n, bins, patches = ax.hist(y_train, bins=bins_num)
        n, bins = np.histogram(y_train, bins=bins_num)
        grouped = [np.where((bins[i] <= y_train) & (y_train < bins[i + 1]))[0] for i in range(len(bins) - 1)]
        sample_indexes = np.array(
            [np.random.choice(grouped_indexes, sample_count) for grouped_indexes in grouped if
             len(grouped_indexes) != 0]).flatten()
        return sample_indexes

    from sklearn.utils import shuffle

    indexes = shuffle(resample_data(y_train, bins_num))

    from sklearn.model_selection import train_test_split
    indexes_train, indexes_valid = train_test_split(indexes, test_size=test_size)

    def iterate(indexes):
        x_train_batch = []
        y_train_batch = []
        while True:
            for index in indexes:
                train_set = train_sets[index]
                image = cv2.imread(train_set.path)
                if train_set.direction < 0:
                    image = np.fliplr(image)

                y_train_batch.append(train_set.y_label)
                x_train_batch.append(image)

                if len(y_train_batch) == batch_size:
                    yield np.array(x_train_batch), y_train_batch
                    x_train_batch = []
                    y_train_batch = []

    return iterate(indexes_train), iterate(indexes_valid), int(len(indexes_train) / batch_size), y_train[indexes]


def show_histgram(y_label, bins_num):
    print(y_label)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    n, bins, patches = ax.hist(y_label, bins=bins_num)
    ax.set_xlabel('rotate')
    ax.set_ylabel('freq')

    fig.show()
    plt.show()


def train(train_generator, valid_generator, samples_per_epoch, validation_size, epoch_count):
    import keras
    from keras.models import Sequential
    from keras.layers import Flatten, Dense
    from keras.layers import Lambda
    from keras.layers import Conv2D
    from keras.layers import MaxPooling2D
    from keras.layers import Cropping2D
    from keras.layers import Dropout

    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    # model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))

    # model.add(Conv2D(20, kernel_size=5, strides=1, activation='relu', input_shape=(160, 320, 3)))
    # model.add(MaxPooling2D(2, strides=2))
    model.add(Conv2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Conv2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Conv2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(Dropout(0.5))

    # model.add(MaxPooling2D())

    # model.add(Conv2D(50, kernel_size=5, strides=1, activation='relu'))
    # model.add(MaxPooling2D(2, strides=2))
    # model.add(Conv2D(6, 5, 5, activation='relu'))
    # model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    # for i in range(samples_per_epoch):
    #     print(i)
    #     print(train_generator.__next__())
    # model.fit_generator(train_generator, validation_split=0.2, shuffle=True, nb_epoch=3, batch_size=10)

    ### add for TensorBoard
    tbcb = keras.callbacks.TensorBoard(log_dir="../tflog/")
    ###

    model.fit_generator(train_generator, validation_data=valid_generator,
                        samples_per_epoch=samples_per_epoch,
                        nb_val_samples=validation_size,
                        nb_epoch=epoch_count,
                        callbacks=[tbcb],
                        verbose=1)

    model.save('model.h5')


def main():
    bins_num = 100
    batch_size = 256
    sample_count = 1000
    test_size = 0.2
    validation_size = 1000
    epoch_count = 40
    train_generator, valid_generator, samples_per_epoch, y_train = get_train_generator(bins_num=bins_num,
                                                                                       batch_size=batch_size,
                                                                                       sample_count=sample_count,
                                                                                       test_size=test_size)
    # show_histgram(y_train, bins_num)
    train(train_generator=train_generator, valid_generator=valid_generator, samples_per_epoch=samples_per_epoch,
          validation_size=validation_size, epoch_count=epoch_count)


main()
