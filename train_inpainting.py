import os
from pathlib import Path
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.python.data import AUTOTUNE
from hrnet import hrnet_keras
import logging
import sys

from lib.hp_optimization import hp_optimization
import argparse
from meanIoU import MeanIoU

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--global_layer', default=False, help="Set true for global layer", action='store_true')
    parser.add_argument('--model_name', default='hrnet', type=str,
                        help='architecture')
    parser.add_argument('--num_epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-lr', '--learning_rate', default=0.1, type=float,
                        help='initial learning rate', dest='lr')
    parser.add_argument('--batch-size', default=4, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--hp_optimization', default=True, help='Do hyper paramater optimization', action='store_true')

    args = parser.parse_args()
    if args.global_layer == True:
        args.model_name = "hrnet-global"
    return args

def load_image(path: str):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3)
    img = tf.cast(img, tf.float32)
    img = tf.image.resize_with_crop_or_pad(img, 256, 256)
    img.set_shape((256, 256, 3))
    img /= 255
    return img


def create_random_ff_mask(img, max_angle=10, max_len=35, max_width=25, times=10):
    """Generate a random free form mask with configuration.
    Args:
        config: Config should have configuration including IMG_SHAPES,
            VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
    Returns:
        tuple: (top, left, height, width)
    """
    height = img.shape[0]
    width = img.shape[1]
    mask = np.zeros((height, width), np.float32)
    times = np.random.randint(times - 5, times)
    for i in range(times):
        start_x = np.random.randint(width)
        start_y = np.random.randint(height)
        for j in range(1 + np.random.randint(5)):
            angle = 0.01 + np.random.randint(max_angle)
            if i % 2 == 0:
                angle = 2 * 3.1415926 - angle
            length = 5 + np.random.randint(max_len - 10, max_len)
            brush_w = 1 + np.random.randint(max_width - 5, max_width)
            end_x = (start_x + length * np.sin(angle)).astype(np.int32)
            end_y = (start_y + length * np.cos(angle)).astype(np.int32)
            cv2.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
            start_x, start_y = end_x, end_y

    return tf.add(tf.multiply(img, tf.expand_dims((1 - mask), -1)), tf.expand_dims(mask,-1))


def load_dataset(path: str, batch_size: int = 8, shuffle: bool = False, pattern: str = "*.jpg"):
    image_paths = list(map(str, Path(path).rglob(pattern)))
    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)

    if shuffle:
        path_ds = path_ds.shuffle(10 * batch_size)

    image_ds = path_ds.map(load_image)
    noisy_ds = image_ds.map(lambda img: create_random_ff_mask(img))

    if batch_size > 0:
        image_ds = image_ds.batch(batch_size)
        noisy_ds = noisy_ds.batch(batch_size)
    # for images in image_ds:
    image_ds = image_ds.cache().prefetch(AUTOTUNE)
    noisy_ds.cache().prefetch(AUTOTUNE)
    return tf.data.Dataset.zip((noisy_ds, image_ds))


def get_model(args):
    hrnet = hrnet_keras(global_layer = args.global_layer)
    print(hrnet.summary())
    return hrnet



def train(args, train_ds, val_ds):
    model = get_model(args)

    if args.hp_optimization:
        best_batch_size, best_lr, best_optimizer = hp_optimization(args, train_ds, val_ds, inpaint=True)
        optimizer = getattr(tf.keras.optimizers, best_optimizer)(learning_rate=best_lr)
        args.batch_size = best_batch_size
        print(f"Best Optimizer : {best_optimizer}, Best Batch Size : {args.batch_size} Best lr : {best_lr}")
    else:
        optimizer = getattr(tf.keras.optimizers, "SGD")(learning_rate=args.lr, momentum=0.9)

    log_dir = f"logs/inpaint/{args.model_name}/"
    Path(log_dir).mkdir(parents=True)
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.FileHandler(filename=log_dir + "settings_log.txt", encoding='utf-8'))
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    callback_list = tf.keras.callbacks.CallbackList([tensorboard_callback],
                                                    add_history=True, model=model)

    mean_iou = MeanIoU(2, 0.4)
    model.compile(optimizer=optimizer, loss='mse', metrics=[mean_iou])

    logging.info(model.summary())
    min_val_loss = 99999

    @tf.function
    def train_step(noise_ds, image_ds, training=True):
        with tf.GradientTape() as tape:
            prediction = model(noise_ds)
            loss = tf.reduce_mean((prediction - image_ds) ** 2)
        if training:
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss

    # running the training loop
    logs = {}
    callback_list.on_train_begin(logs=logs)
    for epoch in range(args.num_epochs):
        callback_list.on_epoch_begin(epoch, logs)

        losses = []
        for i, (noise_ds, image_ds) in enumerate(train_ds):
            loss = train_step(noise_ds, image_ds, training=True)
            losses.append(loss)
            logging.info(f"Epoch : [{epoch + 1} => {i}], loss = {loss:.3f}")
            logs = {'train_loss': np.mean(losses)}

        losses = []
        for i, (noise_ds, image_ds) in enumerate(val_ds):
            loss = train_step(noise_ds, image_ds, training=False)
            losses.append(loss)
        val_logs = {'val_loss': np.mean(losses)}
        logging.info(f"Epoch {epoch}: {logs | val_logs}")
        callback_list.on_epoch_end(epoch, logs=logs | val_logs)

    model.save_weights(f"./models/inpaint/{args.model_name}/{args.model_name}")



def main():
    args = parse_args()
    train_ds = load_dataset(path='./datasets/Paris_StreetView_Dataset/train', shuffle=False, batch_size=4)
    val_ds = load_dataset(path='./datasets/Paris_StreetView_Dataset/val', shuffle=False, batch_size=4, pattern="*.png")
    train(args, train_ds, val_ds)

if __name__ == '__main__':
    main()

