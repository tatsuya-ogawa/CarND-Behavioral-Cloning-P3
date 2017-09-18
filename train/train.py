import csv
import cv2
import os
import numpy as np


class TrainData:
    def __init__(self, y_label, path, direction):
        """
        :param y_label: angle value
        :param path: image file path
        :param direction: direction to fip(1:origin , -1:flip)
        """
        self.y_label = y_label
        self.path = path
        self.direction = direction


def get_train_generator(sample_count, batch_size, bins_num, test_size):
    """
    :param sample_count: resample count
    :param batch_size: batch size of generator
    :param bins_num: resample bins count
    :param test_size: test data size of train_test_split
    :return:  train_generator, valid_generator, samples_per_epoch, y_all_data, indexes_train
    """
    lines = []
    data_dir = os.path.join('..', 'drive_data')
    with open(os.path.join(data_dir, 'driving_log.csv')) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    train_sets = []
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

    ### resample indexes to balancing data
    def resample_data(y_train, bins_num):
        n, bins = np.histogram(y_train, bins=bins_num)
        grouped = [np.where((bins[i] <= y_train) & (y_train < bins[i + 1]))[0] for i in range(len(bins) - 1)]
        sample_indexes = np.array(
            [np.random.choice(grouped_indexes, sample_count) for grouped_indexes in grouped if
             len(grouped_indexes) != 0]).flatten()
        return sample_indexes

    ### train and validation data split
    from sklearn.utils import shuffle
    indexes = shuffle(resample_data(y_train, bins_num))
    from sklearn.model_selection import train_test_split
    indexes_train, indexes_valid = train_test_split(indexes, test_size=test_size)

    def get_generator_from_indexes(indexes):
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

    return get_generator_from_indexes(indexes_train), get_generator_from_indexes(indexes_valid), int(
        len(indexes_train) / batch_size), y_train, indexes_train


def show_histgram(y_all_data, indexes_train, bins_num):
    """
    :param y_all_data: original angle data
    :param indexes_train: resampled indexes
    :param bins_num: bins num of resampling
    """
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.hist(y_all_data, bins=bins_num)
    ax1.set_title('Original Data')

    ax2.hist(y_all_data[indexes_train], bins=bins_num)
    ax2.set_title('Resampled Data')

    fig.show()
    plt.savefig('data_histogram.png')


def train(train_generator, valid_generator, samples_per_epoch, validation_size, epoch_count):
    import keras
    from keras.models import Sequential
    from keras.layers import Flatten, Dense
    from keras.layers import Lambda
    from keras.layers import Conv2D
    from keras.layers import Cropping2D
    from keras.layers import Dropout

    ### Define model
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Conv2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Conv2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Conv2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    ###

    ### Save model and model figure(require pydot-ng and graphiviz is installed)
    from keras.utils.visualize_util import plot
    plot(model, show_shapes=True, to_file='model.png')
    with open('model.json',mode='w') as f:
        f.write(model.to_json())
    ###

    ### add for TensorBoard
    tbcb = keras.callbacks.TensorBoard(log_dir="../tflog/")
    ###

    ### Learning
    model.fit_generator(train_generator, validation_data=valid_generator,
                        samples_per_epoch=samples_per_epoch,
                        nb_val_samples=validation_size,
                        nb_epoch=epoch_count,
                        callbacks=[tbcb],
                        verbose=1)
    ###

    ### Save learned weights
    model.save('model.h5')
    ###


def main():
    bins_num = 100
    batch_size = 256
    sample_count = 1000
    test_size = 0.2
    validation_size = 1000
    epoch_count = 30
    train_generator, valid_generator, samples_per_epoch, y_all_data, indexes_train = get_train_generator(
        bins_num=bins_num,
        batch_size=batch_size,
        sample_count=sample_count,
        test_size=test_size)
    show_histgram(y_all_data, indexes_train, bins_num)

    train(train_generator=train_generator, valid_generator=valid_generator, samples_per_epoch=samples_per_epoch,
          validation_size=validation_size, epoch_count=epoch_count)


main()
