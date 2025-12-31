import numpy as np
import cv2

def onehot_encoding_2d(label, num_classes):
    shape = label.shape
    onehot = np.zeros((num_classes, shape[0], shape[1]))

    for i in range(num_classes):
        indices = np.where(label == i)
        onehot[i][indices] = 1.0
    return onehot

def window_CT(slice, min=-100, max=300):
      img = np.clip(slice, min, max)
      img = (img-min)/(max-min)
      img_slice_resized = cv2.resize(
        img, (512, 512),
        interpolation=cv2.INTER_LINEAR)
      return img_slice_resized

def to1channel(x):
    batch_size = x.size(0)
    num_slices = x.size(1)
    x = x.reshape(batch_size * num_slices, 1, x.size(2), x.size(3))
    return x, batch_size, num_slices

def tonchannel(x, batch_size, num_slices):
    x = x.reshape(batch_size, num_slices, -1)
    return x
