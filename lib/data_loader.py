import tensorflow as tf
from tensorflow.keras.datasets import cifar10, cifar100
import numpy as np



def create_data(args, batch_size):
    if args.dataset in ['CIFAR-10', 'CIFAR-100']:
        n_channels, n_size = 3, 32
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        if args.dataset == 'CIFAR-10':
            n_class = 10
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            # (50000,32,32,3), (50000,10)

        elif args.dataset == 'CIFAR-100':
            n_class = 100
            (x_train, y_train), (x_test, y_test) = cifar100.load_data()

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        # normalize the data
        for channel in range(3):
            x_train[:, :, :, channel] = (x_train[:, :, :, channel] - mean[channel]) / std[channel]
            x_test[:, :, :, channel] = (x_test[:, :, :, channel] - mean[channel]) / std[channel]

        y_train = tf.keras.utils.to_categorical(y_train, n_class)
        y_test = tf.keras.utils.to_categorical(y_test, n_class)

        train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            horizontal_flip=True,
            width_shift_range=0.1,
            height_shift_range=0.1,
            validation_split=0.2,
        )
        test_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            horizontal_flip=True
        )
        train_gen.fit(x_train)
        train_data = train_gen.flow(
            x_train,
            y_train,
            batch_size=args.batch_size,
            seed=1234,
            shuffle=True,
            subset="training"
        )

        val_data = train_gen.flow(
            x_train,
            y_train,
            # batch_size=args.batch_size,
            batch_size=batch_size,
            seed=1234,
            shuffle=False,
            subset="validation"
        )
        test_data = test_gen.flow(
            x_test,
            y_test,
            # batch_size=args.batch_size,
            batch_size=batch_size,
            seed=1234,
            shuffle=False
        )

    else:
        print('Not supported currently.')
        exit(1)
    return train_data, val_data, test_data, n_class