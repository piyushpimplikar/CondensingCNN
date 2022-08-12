
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, ReLU
import shutil
from math import cos, pi

class Hswish(tf.Module):
    def __init__(self):
        super(Hswish, self).__init__()

    def __call__(self, x):
        return x * ReLU(x + 3.) / 6.

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class h_sigmoid(tf.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = ReLU()

    def __call__(self, x):
        return self.relu(x + 3) / 6


class h_swish(tf.Module):
    def __init__(self):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid()

    def __call__(self, x):
        return x * self.sigmoid(x)

class SeparableConv2d(tf.Module):
    def __init__(self,
                 x,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding="valid",
                 dilation=1,
                 bias=True,
                 padding_mode='zeros',
                 depth_multiplier=1,
        ):
        super().__init__()
        
        intermediate_channels = int( in_channels * depth_multiplier )
        self.spatialConv = Conv2D(
             filters=intermediate_channels,
             kernel_size=(kernel_size,kernel_size),
             strides=(stride,stride),
             padding=padding,
             dilation_rate=dilation,
             groups=in_channels,
             use_bias=bias
        )
        self.pointConv = Conv2D(
             out_channels=out_channels,
             kernel_size=(1,1),
             strides=(1,1),
             padding="valid",
             dilation_rate=1,
             use_bias=bias
        )
    
    def __call__(self, x):
        return self.pointConv(self.spatialConv(x))



class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        # mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


def get_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    # with torch.no_grad():
    maxk = max(topk)
    batch_size = target.shape[0]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", logger=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logger = logger

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        message = '\t'.join(entries)
        #print(message)
        if self.logger:
            self.logger.info(message)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def save_checkpoint(state, is_best, EXP_NAME='', filename='-checkpoint.h5'):
    state.save_weights(EXP_NAME + filename)
    if is_best:
        shutil.copyfile(EXP_NAME + filename, EXP_NAME + '-model_best.h5.tar')


class CrossEntropyLabelSmooth(tf.Module):
    def __init__(self, num_classes=1000, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = tf.LogSoftmax(dim=1)

    def __call__(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = tf.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def adjust_learning_rate(optimizer, epoch, iteration, num_iter, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = optimizer.param_groups[0]['lr']

    warmup_epoch = 5 if args.warmup else 0
    warmup_iter = warmup_epoch * num_iter
    current_iter = iteration + epoch * num_iter
    max_iter = args.epochs * num_iter

    if args.lr_decay == 'step':
        lr = lr * (0.1 ** (epoch // 30))
    elif args.lr_decay == 'cos':
        lr = args.learning_rate * (1 + cos(pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2
    else:
        raise ValueError('Unknown lr mode{} : '.format(args.lr_decay))

    if epoch < warmup_epoch:
        lr = args.learning_rate * current_iter / warmup_iter

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = tf.Variable(tf.Variable(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x
