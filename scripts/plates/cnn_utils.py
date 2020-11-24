import numpy as np

def confidence(oh):
    return np.max(oh) / np.sum(oh)


# Constructs a one-hot vector
def one_hot(size, index):
    oh = np.zeros(size, dtype=float)
    oh[index] = 1
    return oh

def one_hot_to_number(oh):
    index = np.argmax(oh)
    return chr(ord("0") + index)

def one_hot_to_char(oh):
    index = np.argmax(oh)
    return chr(ord("A") + index)

def one_hot_to_spot_number(oh):
    index = np.argmax(oh)
    return chr(ord("1") + index)


# takes a list of channel-less (e.g. grayscale) images
# and returns list of images with explicitly one channel
def reshape(single_channel):
    if len(single_channel.shape) == 4:
        return single_channel
    return np.reshape(single_channel, single_channel.shape + (1,))
