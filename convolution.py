"""
Contents:
    Custom Convolutions
TO DO:
    Max and Average Pooling

@vbar
"""
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt


# sample import_fn
def input_fn():
    # change to import your image
    return misc.ascent()


def show_img(image):
    """ returns input image """
    plt.grid(False)
    plt.gray()
    plt.axis('off')
    plt.imshow(image)
    plt.show()


def convolution(image: np.ndarray, conv: np.ndarray, channels_last: bool = True,
                brightness: float = 1., bias: float = 0.):
    """
    Given a filter matrix, it returns a convolution of an image.
    Note :: The purpose of this function is to show how convolutions are computed
    in a simple and comprehensive way. Thus, performance is sacrificed for this purpose.

    :param image: numpy array of the image convolved
    :param conv: numpy array with the filter matrix.
                 It needs to be a square with odd dimensions matrix.
    :param channels_last: Change to False if image channels are first
    :param brightness: float to adjust brightness using softmax function.
    :param bias: float to adjust "contrast".
                 For example, high bias with low brightness returns emboss effect.
    :return: filtered image
    """
    double_dim = False

    if image.ndim is 2:
        image = np.expand_dims(image, axis=-1)
        double_dim = True
    elif image.ndim is 3 and not channels_last:
        image = np.moveaxis(image, 0, 2)
    else:
        raise Exception(f"Image dims should be either 2 or 3 : {image.ndim}")

    if conv.ndim is 2:
        x_conv, y_conv = conv.shape
    else:
        raise Exception(f"Conv dims should be equal with 2 : {conv.ndim}")
    if not (x_conv == y_conv or x_conv % 2):
        raise Exception(f"Conv should be a square matrix : {conv.shape}")

    # softmax function to adjust brightness
    if np.sum(conv) != brightness:
        def normalizer(vector):
            base_map = lambda x: np.e ** x
            denominator = np.sum(np.vectorize(base_map)(vector))
            norm_map = lambda x: (base_map(x) * brightness) / denominator
            vector = np.vectorize(norm_map)(vector)
            return vector
        conv = normalizer(conv)

    x_axis, y_axis, channel = image.shape
    filtered = np.copy(image)
    # loop through pixel "centers"
    for xi in range(x_conv // 2, x_axis - x_conv // 2):
        for yi in range(y_conv // 2, y_axis - y_conv // 2):
            color = np.zeros(channel)
            # loop through convolution coordinates
            for xc in range(x_conv):
                for yc in range(y_conv):
                    x_idx = xi - x_conv // 2 + yc
                    y_idx = yi - y_conv // 2 + xc
                    # set filter values for each color
                    for c in range(color.ndim):
                        color[c] += image[x_idx, y_idx, c] * conv[xc, yc]
            # apply and adjust offset color values
            for c in range(color.ndim):
                filtered[xi, yi, c] = min(max(int(color[c] + bias), 0), 255)

    # input had 2 dims or channels first
    if double_dim:
        filtered = np.squeeze(filtered, axis=-1)
    if not channels_last:
        filtered = np.moveaxis(filtered, 2, 0)

    return filtered


if __name__ == '__main__':
    # input image and sample filter
    img = input_fn()
    my_filter = np.array([[-1, 1, -1], [1, -2, 1], [-1, 2, 0]], dtype=float)

    # compute convolution
    new_img = convolution(img, my_filter, brightness=1, bias=100)

    # show filtered image
    show_img(new_img)
