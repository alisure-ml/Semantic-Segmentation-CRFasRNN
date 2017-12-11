import numpy as np
from PIL import Image


# 读数据、减均值、pad大小到500，返回处理后的数据和原始大小
def get_preprocessed_image(file_name):
    mean_values = np.array([123.68, 116.779, 103.939], dtype=np.float32)  # RGB mean values
    mean_values = mean_values.reshape(1, 1, 3)
    im = np.array(Image.open(file_name)).astype(np.float32)
    im = im - mean_values
    im = im[:, :, ::-1]
    img_h, img_w, img_c = im.shape
    if img_h > 500 or img_w > 500:
        raise ValueError("Please resize your images to be not bigger than 500 x 500.")

    pad_h = 500 - img_h
    pad_w = 500 - img_w
    im = np.pad(im, pad_width=((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
    return im.astype(np.float32).reshape(1, 500, 500, 3), img_h, img_w


# 输入softmax,经过argmax后输出着色结果
def get_label_image(probs, img_h, img_w):
    labels = probs.argmax(axis=2).astype("uint8")[:img_h, :img_w]
    label_im = Image.fromarray(labels, "P")
    # Pascal VOC color palette for labels
    _PALETTE = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128,
                64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 64, 128, 128, 192, 128, 128,
                0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 128, 64, 128, 0, 192, 128, 128, 192, 128,
                64, 64, 0, 192, 64, 0, 64, 192, 0, 192, 192, 0]
    label_im.putpalette(_PALETTE)
    return label_im
