from dataclasses import dataclass
import itertools
from datetime import datetime
from tensorboard.plugins.hparams import api as hp
import tensorflow as tf
from pathlib import Path
import os
from model import pdenet, resnet32, resnet_original, pdenet_original
from hrnet import hrnet_keras
from meanIoU import MeanIoU

@dataclass
class Settings:
    name: str
    optimizer: str
    batch_size: int
    lr: float


def get_architecture_for_dataset(args, n_class=10, aux=True):
    global_ft = '-Global' in args.model
    if args.dataset in ['CIFAR-10', 'CIFAR-100']:
        if args.model == 'Resnet':
            if args.original:
                cnn = resnet_original( num_classes=n_class, m=args.resnet_m, n1=args.n1, n2=args.n2, n3=args.n3, n4=args.n4, args=args )
            else:
                cnn = resnet32(num_classes=n_class, m=args.resnet_m, n1=args.n1, n2=args.n2, n3=args.n3, n4=args.n4, args=args )
        elif args.model == 'Resnet-Global':
            if args.original:
                cnn = pdenet_original( num_classes=n_class, m=args.resnet_m, n1=args.n1, n2=args.n2, n3=args.n3, n4=args.n4, args=args )
            else:
                cnn = pdenet(num_classes=n_class, progress=True, m=args.resnet_m, n1=args.n1, n2=args.n2, n3=args.n3, n4=args.n4, args=args )
        else:
            raise ValueError('Incorrect model architecture.')
    else:
        raise ValueError('Dataset not supported.')
    return cnn

def get_model(args, batch_size, n_class):
    input = tf.keras.Input(shape=(32,32,3))
    cnn = get_architecture_for_dataset(args, n_class, aux=True)
    output = cnn(input)
    model = tf.keras.Model(input, output)

    return model

def hyper_parameters():

    optimizers = ["Adam","SGD"]
    batch_size = [32, 64, 128, 256]
    lr = ["1e-2","1e-3"]

    # optimizers = ["Adam"]
    # batch_size = [32]
    # lr = ["1e-2"]

    hyperparams_list = [optimizers, batch_size, lr]
    hyperparms = list(itertools.product(*hyperparams_list))

    return hyperparms

def hp_optimization(args, train_data, val_data, n_class=10, inpaint=False):
    print("==============================Finding Best Hyper Paramaters========================================")
    hyper_params = hyper_parameters()

    min_val_loss = 99999999
    best_batch_size = 0
    best_lr = 0
    best_optimizer=""
    if inpaint:
        log_dir = f"./logs/inpaint/{args.model_name}/hparam_tuning"
    else:
        log_dir = f"./logs/{args.dataset}/{args.model_name}/hparam_tuning"
    if not os.path.exists(log_dir):
        Path(log_dir).mkdir(parents=True)
    for hyper_param in hyper_params:
        name = str(hyper_param[0]) + "_" + str(hyper_param[1]) + "_" + str(
            hyper_param[2]) + f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        setting = Settings(name=name, optimizer=hyper_param[0], batch_size=hyper_param[1], lr=float(hyper_param[2]))
        optimizer = getattr(tf.keras.optimizers, setting.optimizer)(learning_rate=setting.lr)
        print(setting)

        if inpaint:
            mean_iou = MeanIoU(2, 0.4)
            hp_model = hrnet_keras(global_layer = args.global_layer)
            hp_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[mean_iou])
        else:
            hp_model = get_model(args, setting.batch_size, n_class)
            hp_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


        #check which hyper parameter is best for the 1st epoch and continue training with those hyper parameters.
        history = hp_model.fit(train_data,
                            validation_data=val_data,
                            batch_size=setting.batch_size,
                            epochs=1,
                            steps_per_epoch=len(train_data) - 1,
                            validation_steps=len(val_data) - 1
                            )

        if history.history["val_loss"][-1] < min_val_loss:
            min_val_loss = history.history["val_loss"][-1]
            best_model = hp_model
            best_batch_size = setting.batch_size
            best_lr = setting.lr
            best_optimizer = setting.optimizer
            print("Min loss ", min_val_loss)

        with tf.summary.create_file_writer(log_dir).as_default():
            hp.hparams({"optimizer": setting.optimizer, "batch_size": setting.batch_size, "lr":setting.lr})
            tf.summary.scalar("val_loss", history.history["val_loss"][-1], step=1)

    return best_batch_size, best_lr, best_optimizer



