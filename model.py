import tensorflow as tf
import tensorflow.contrib.layers as tcl
from crf_rnn_layer_tf import CrfRnnLayer


class CRFasRNN(object):

    def __init__(self, height, weight, num_class):
        self.num_class = num_class
        self.img_input = tf.placeholder(dtype=tf.float32, shape=[None, height, weight, 3])
        pass

    def build_net(self, img_input):
        # FCN model
        conv = tcl.conv2d(img_input, 128, kernel_size=3, stride=5, padding="SAME")  # [100, 100, 128]
        output = tcl.conv2d_transpose(conv, self.num_class, kernel_size=3, stride=5, padding="SAME")  # [500, 500, 21]

        # 添加CRF后端：必须满足output和img_input的大小相同
        output = CrfRnnLayer(num_class=self.num_class, theta_alpha=160.,
                             theta_beta=3., theta_gamma=3., num_iter=10)(output, img_input)
        # 添加CRF后端
        return output

    pass
