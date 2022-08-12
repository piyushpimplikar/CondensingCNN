#!/usr/bin/env python3
import argparse
import logging
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler
from model import pdenet, resnet32, resnet_original, pdenet_original
import warnings
from lib.hp_optimization import hp_optimization
warnings.filterwarnings("ignore")
from lib.data_loader import create_data
from lib.logging_setup import logging_setup
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--K', default=5, type=int,
                        metavar='K', help='Number of iterations in the Global feature extractor block (default: 3)')
    parser.add_argument('--non_linear', default=True, action='store_true')
    parser.add_argument('--pde_state', default=0, type=int,
                        metavar='N', help='PDE State so that we can try out multiple(default: 0)')
    parser.add_argument('-dxy', '--constant_Dxy', default=False, action='store_true')
    parser.add_argument('--use_silu', default=False, action='store_true')
    parser.add_argument('--custom_uv', default='', type=str)
    parser.add_argument('--custom_dxy', default='', type=str)
    parser.add_argument('--use_res', default=False, action='store_true')
    parser.add_argument('--init_h0_h', default=False, action='store_true')
    parser.add_argument('--use_f_for_g', default=False, action='store_true')
    parser.add_argument('--old_style', default=False, action='store_true')
    parser.add_argument('-nof', '--no_f', default=False, action='store_true')
    parser.add_argument('--dt', type=float, default=0.2, help='Random erase prob (default: 0.)')
    parser.add_argument('--dx', type=int, default=1, help='Random erase prob (default: 0.)')
    parser.add_argument('--dy', type=int, default=1, help='Random erase prob (default: 0.)')
    parser.add_argument('--cDx', type=float, default=1., help='Random erase prob (default: 0.)')
    parser.add_argument('--cDy', type=float, default=1., help='Random erase prob (default: 0.)')

    parser.add_argument('--no_diffusion', default=False, help='Enable Diffusion', action='store_true')
    parser.add_argument('--no_advection', default=False, help='Enable advection', action='store_true')

    parser.add_argument('-ct', '--cutout', default=False, action='store_true')
    parser.add_argument('-o', '--original', default=False, action='store_true')
    parser.add_argument('-r', '--restart', default=False, action='store_true')
    parser.add_argument('-rk', '--restart_known', default=False, action='store_true')
    parser.add_argument('-rmn', '--restart_model_name', type=str, help='model to restart from')
    parser.add_argument('--warmup', action='store_true', help='set lower initial learning rate to warm up the training.')
    parser.add_argument('--separable', action='store_true', help='using separable convolutions.')
    parser.add_argument('-dc', '--lr_decay', default='cos', type=str)
    parser.add_argument('-p', '--logging.info-freq', default=500, type=int,
                        metavar='N', help='logging.info frequency (default: 10)')
    parser.add_argument('-m', '--resnet-m', default=1, type=int,
                        metavar='N', help='Number of repeats in one resnet block (default: 2)')
    parser.add_argument('-wdt', '--width', default=2, type=int,
                        metavar='N', help='Width wide resnet block (default: 2)')
    parser.add_argument('-ds', '--dataset', default='CIFAR-10', type=str,
                        help='dataset ( CIFAR-10/CIFAR-100/Imagenet-1000 )')
    parser.add_argument('--drop', type=float, default=0.2, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--model', default='Resnet-Global', type=str,
                        help='architecture ( Resnet-Global/CIFAR )')

    parser.add_argument('--model_name', default='Resnet-Global', type=str,
                        help='architecture ( Resnet-Global/CIFAR )')

    parser.add_argument('-d', '--data', default='./data',
                       type=str, metavar='DIR', help='path to dataset')
    parser.add_argument('-n', '--n_holes', default=1, type=int, metavar='N',
                        help='number of holes in cutout augmentation')
    parser.add_argument('-l', '--length', default=16, type=int, metavar='N',
                        help='length of each hole in cutout augmentation')
    parser.add_argument('-e', '--epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-ek', '--efficient_k', default=0, type=int, metavar='N',
                        help='Efficient Net variant (0-8)')

    parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "sgd"')
    parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: None, use opt default)')
    parser.add_argument('-lr', '--learning_rate', default=0.1, type=float,
                        help='initial learning rate', dest='lr')
    parser.add_argument('-wd', '--weight-decay', default=6e-5, type=float,
                        metavar='W', help='weight decay (default: 5e-5)',
                        dest='weight_decay')

    parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                        help='Random erase prob (default: 0.)')
    parser.add_argument('--remode', type=str, default='const',
                        help='Random erase mode (default: "const")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='step', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "step"')
    #parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
    #                    help='learning rate (default: 0.01)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                        help='learning rate cycle len multiplier (default: 1.0)')
    parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                        help='learning rate cycle limit')
    parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                        help='warmup learning rate (default: 0.0001)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    #parser.add_argument('--epochs', type=int, default=200, metavar='N',
    #                    help='number of epochs to train (default: 2)')
    parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                        help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
    parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')


    parser.add_argument('--amp', action='store_true', default=False,
                        help='use NVIDIA Apex AMP or Native AMP for mied precision training')

    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('-tb', '--test-batch-size', default=512, type=int,
                        metavar='N',
                        help='[test] mini-batch size (default: 512)')

    parser.add_argument('--n1', default=16, type=int,
                        help='number of total filters in first convolutional layer (CIFAR-10/100)')
    parser.add_argument('--n2', default=32, type=int,
                        help='number of total filters in second convolutional layer (CIFAR-10/100)')
    parser.add_argument('--n3', default=64, type=int,
                        help='number of total filters in third convolutional layer (CIFAR-10/100)')
    parser.add_argument('--n4', default=64, type=int,
                        help='number of total filters in third convolutional layer (CIFAR-10/100)')


    # Model Exponential Moving Average
    parser.add_argument('--model-ema', action='store_true', default=False,
                        help='Enable tracking moving average of model weights')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                        help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
    parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                        help='decay factor for model weights moving average (default: 0.9998)')

    parser.add_argument('--log_dir', type=str, default='./logs/', help='Log Directory')
    parser.add_argument('--model_dir', type=str, default='./models/', help='Model Checkpoint Directory')
    parser.add_argument('--hp_optimization', default=False, help='Do hyper paramater optimization', action='store_true')

    args = parser.parse_args()
    return args


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

def get_model(args, n_class, logger):
    input = tf.keras.Input(shape=(32,32,3))
    cnn = get_architecture_for_dataset(args, n_class, aux=True)
    output = cnn(input)
    model = tf.keras.Model(input, output)

    if logger is not None:
        logger.info(model.summary())

    return model

def decay_schedule(epoch, lr):
    # decay by 0.1 every 5 epochs; use `% 1` to decay after each epoch
    if (epoch == 150) or (epoch == 225):
        lr = lr * 0.1
    return lr

def plot(args, history):
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(20, 10)
    epochs = [i for i in range(args.epochs)]
    ax[0].plot(epochs, history.history["accuracy"], label="Train")
    ax[0].plot(epochs, history.history["val_accuracy"], label="Val")
    ax[0].legend()
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Accuracy")
    ax[0].set_title("Epoch vs Accuracy")

    ax[1].plot(epochs, history.history["loss"], label="Train")
    ax[1].plot(epochs, history.history["val_loss"], label="Val")
    ax[1].legend()
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Loss")
    ax[1].set_title("Epoch vs Loss")

    plt.savefig(f"./logs/{args.dataset}-{args.model_name}.png")



def train(args, logger, train_data, val_data, test_data, n_class):
    criterion = tf.keras.losses.CategoricalCrossentropy()
    model = get_model(args, n_class, logger)
    if args.hp_optimization:
        best_batch_size, best_lr, best_optimizer = hp_optimization(args, train_data, val_data, n_class)
        cnn_optimizer = getattr(tf.keras.optimizers, best_optimizer)(learning_rate=best_lr)
        args.batch_size = best_batch_size
        logger.info(f"Best Optimizer : {cnn_optimizer}, Best Batch Size : {args.batch_size} Best lr : {best_lr}")
    else:
        cnn_optimizer = getattr(tf.keras.optimizers, "SGD")(learning_rate=args.lr, momentum=0.9)



    model.compile(loss = criterion, optimizer = cnn_optimizer, metrics=['accuracy'])

    earlyStopping = tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=4,
                        restore_best_weights=True
                    )

    log_path = f"./logs/{args.dataset}/{args.model_name}"
    tensorboard = TensorBoard(log_dir=log_path)
    checkPoint = tf.keras.callbacks.ModelCheckpoint(filepath = log_path + "/checkpoints/" + "cp-{epoch:04d}.ckpt",
                                                        save_best_only = True,
                                                        mode = 'min',
                                                        monitor = 'val_loss')
    logs_csv = tf.keras.callbacks.CSVLogger(log_path + "/logs.csv", separator=',', append=False)
    lr_scheduler = LearningRateScheduler(decay_schedule)

    print("==============================Training Started========================================")

    # model.load_weights(f"./models/{args.dataset}/{args.model_name}/{args.model_name}")
    try:
        history = model.fit(train_data,
                    validation_data=val_data,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    steps_per_epoch=len(train_data)-1,
                    validation_steps=len(val_data)-1,
                    callbacks=[
                        checkPoint,
                        tensorboard,
                        lr_scheduler,
                        logs_csv
                    ])

    except KeyboardInterrupt:
        model.save_weights(f"./models/{args.dataset}/{args.model_name}/{args.model_name}_keyboardInterrupted")
    finally:
        model.save_weights(f"./models/{args.dataset}/{args.model_name}/{args.model_name}")

    loss, acc = model.evaluate(test_data)
    logger.info(f"Test Accuracy : {acc}, Test Loss : {loss}")

    plot(args, history)

def model_parameters(args):
    model = get_model(args, n_class=10, logger=None)
    # model.load_weights("./models/CIFAR-10/Resnet-Global/Resnet-Global")
    model.load_weights("./logs/CIFAR-10/Resnet-Global/checkpoints/variables/variables")
    input = tf.random.normal((32, 32, 32, 3))
    print(model.predict(input).shape)
    return

def main():
    args = parse_args()
    # model_parameters(args)
    logger = logging_setup(args)
    train_data, val_data, test_data, n_class = create_data(args, args.batch_size)
    train(args, logger, train_data, val_data, test_data, n_class)


if __name__ == '__main__':
    main()






























# def test(loader, cnn, model_name):
#     batch_time = AverageMeter('Time', ':6.3f')
#     losses = AverageMeter('Loss', ':.4e')
#     top1 = AverageMeter('Acc@1', ':6.2f')
#     top5 = AverageMeter('Acc@5', ':6.2f')
#     progress = ProgressMeter(
#         len(loader),
#         [batch_time, losses, top1, top5],
#         prefix='Test('+ model_name + '): ', logger=logger)
#
#     cnn.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
#     correct = 0.
#     total = 0.
#     end = time.time()
#
#     aux_available = False
#     if args.model not in ['Resnet-Global', 'Resnet', 'Resnet-Global-Res', 'Densenet', 'Densenet-Global', 'Compact', 'WideResnet', 'WideResnet-Global' , 'm_resnet', 'm_odenet', 'm_global' ]:
#         aux_available = True
#
#     for i, (images, labels) in enumerate(loader):
#
#         if not aux_available: #args.model in ['Resnet-Global', 'Resnet']:
#             pred = cnn(images)
#         else:
#             pred, _ = cnn(images)
#
#         loss = criterion(pred, labels)
#
#         # measure accuracy and record loss
#         acc1 = accuracy.update_state(pred,labels)
#         acc1 = acc1.results().numpy()
#
#         # acc1, acc5 = get_accuracy(pred, labels, topk=(1, 5))
#         losses.update(loss.item(), images.size(0))
#         top1.update(acc1[0], images.size(0))
#         # top5.update(acc5[0], images.size(0))
#
#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()
#
#         if i % args.logging.info_freq == 0:
#             progress.display(i)
#
#         pred = tf.max(pred.data, 1)[1]
#         total += labels.size(0)
#         correct += (pred == labels).sum().item()
#     progress.display(i)
#
#     message = ' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5)
#     # this should also be done with the ProgressMeter
#     logging.info(message)
#     logger.info(message)
#
#     val_acc = correct / total
#     return val_acc
#
#
# def main():
#     train()
# if __name__ == '__main__':
#     main()


# def train(args, logger):
#     train_data, val_data, test_data, n_class = create_data(args)
#     model = get_model(args, n_class)
#
#     criterion = tf.keras.losses.CategoricalCrossentropy()
#     cnn_optimizer = tf.keras.optimizers.SGD( learning_rate =args.lr, momentum=0.9, nesterov=True)
#
#     start_epoch = 0
#     best_acc1 = 0.0
#
#     @tf.function
#     def train_step(images, labels, training=True):
#         with tf.GradientTape() as tape:
#             pred = model(images)
#             entropy_loss = criterion(labels, pred)
#
#         if training:
#             grads = tape.gradient(entropy_loss, model.trainable_weights)
#             cnn_optimizer.apply_gradients(zip(grads, model.trainable_weights))
#
#         return entropy_loss, pred
#
#     for epoch in range(start_epoch, args.epochs):
#
#         entropy_loss_avg = 0.
#         correct = 0.
#         total = 0.
#
#         batch_time = AverageMeter('Time', ':6.3f')
#         data_time = AverageMeter('Data', ':6.3f')
#         losses = AverageMeter('Loss', ':.4e')
#         acc = AverageMeter('Acc', ':6.2f')
#         progress = ProgressMeter(
#             len(train_data),
#             [batch_time, data_time, losses, acc],
#             prefix="Epoch: [{}]".format(epoch),
#             logger=logger)
#
#
#         end = time.time()
#         for i, (images, labels) in enumerate(train_data):
#
#             data_time.update(time.time() - end)
#
#
#             entropy_loss, pred = train_step(images, labels, training=True)
#
#             losses.update(entropy_loss, images.shape[0])
#             entropy_loss_avg += entropy_loss
#
#             pred = tf.argmax(pred, axis=1)
#             labels = tf.argmax(labels, axis=1)
#             total += labels.shape[0]
#             correct += tf.reduce_sum(tf.cast(pred == labels, tf.float32))
#             accuracy = correct / total
#             acc.update(accuracy, images.shape[0])
#
#             batch_time.update(time.time() - end)
#             end = time.time()
#
#             if i % args.logging.info_freq == 0:
#                 progress.display(i)
#
#         progress.display(i)
#
#         test_acc = test(test_data, model, model_name='Original')
#
#
#         # remember best acc@1 and save checkpoint
#         acc1 = test_acc
#         is_best = acc1 > best_acc1
#         best_acc1 = max(acc1, best_acc1)
#
#         message = '---- Best so far --- ' + str(best_acc1)
#         logger.info(message)
#
#         model_train_dict = {
#                 'epoch': epoch + 1,
#                 'arch': "GlobalLayer-" + args.model,
#                 'state_dict': model.state_dict(),
#                 'best_acc1': best_acc1,
#                 'optimizer' : cnn_optimizer.state_dict(),
#         }
#
#         save_checkpoint(model_train_dict, is_best, EXP_NAME='./models/' + exp_name)
