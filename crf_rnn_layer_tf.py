import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

custom_module = tf.load_op_library('./cpp/high_dim_filter.so')


# Register gradients for the custom op
@ops.RegisterGradient("HighDimFilter")
def _high_dim_filter_grad(op, grad):
    """
    Gradients for the HighDimFilter op. We only need to calculate the gradients
    w.r.t. the first input (unaries) as we never need to backprop errors to the
    second input (RGB values of the image).

    Args:
    op: The `high_dim_filter` operation that we are differentiating.
    grad: Gradients with respect to the output of the `high_dim_filter` op.

    Returns:
    Gradients with respect to the input of `high_dim_filter`.
    """
    rgb = op.inputs[1]
    grad_vals = custom_module.high_dim_filter(grad, rgb,
                                              bilateral=op.get_attr("bilateral"),
                                              theta_alpha=op.get_attr("theta_alpha"),
                                              theta_beta=op.get_attr("theta_beta"),
                                              theta_gamma=op.get_attr("theta_gamma"),
                                              backwards=True)
    return [grad_vals, tf.zeros_like(rgb)]


class CrfRnnLayer(object):
    """
    Implements the CRF-RNN layer described in:

    Conditional Random Fields as Recurrent Neural Networks,
    S. Zheng, S. Jayasumana, B. Romera-Paredes, V. Vineet, Z. Su, D. Du, C. Huang and P. Torr,
    ICCV 2015
    """

    def __init__(self, num_class, theta_alpha, theta_beta, theta_gamma, num_iter):
        self.num_class = num_class
        self.num_iter = num_iter

        self.theta_alpha = theta_alpha
        self.theta_beta = theta_beta
        self.theta_gamma = theta_gamma

        # Weights of the spatial kernel
        self.spatial_ker_weights = self.get_weights((self.num_class, self.num_class), name='spatial_ker_weights')
        # Weights of the bilateral kernel
        self.bilateral_ker_weights = self.get_weights((self.num_class, self.num_class), name='bilateral_ker_weights')
        # Compatibility matrix
        self.compatibility_matrix = self.get_weights((self.num_class, self.num_class), name='compatibility_matrix_weights')

        pass

    # [net_output, img_input] [[w, h, n_class], [w, h, channel]]
    def __call__(self, net_output, img_input):
        image_shape = net_output.get_shape()
        # 将channel维放到前面
        unaries = tf.transpose(net_output[0, :, :, :], perm=(2, 0, 1))  # [n_class, w, h]
        rgb = tf.transpose(img_input[0, :, :, :], perm=(2, 0, 1))  # [channel, w, h]

        all_ones = np.ones((self.num_class, image_shape[1], image_shape[2]), dtype=np.float32)  # [n_class, w, h]

        # Prepare filter normalization coefficients
        spatial_norm_vals = custom_module.high_dim_filter(all_ones, rgb, bilateral=False, theta_gamma=self.theta_gamma)
        bilateral_norm_vals = custom_module.high_dim_filter(all_ones, rgb, bilateral=True,
                                                            theta_alpha=self.theta_alpha, theta_beta=self.theta_beta)
        q_values = unaries

        for i in range(self.num_iter):
            softmax_out = tf.nn.softmax(q_values, dim=0)

            # Spatial filtering
            spatial_out = custom_module.high_dim_filter(softmax_out, rgb, bilateral=False, theta_gamma=self.theta_gamma)
            spatial_out = spatial_out / spatial_norm_vals

            # Bilateral filtering
            bilateral_out = custom_module.high_dim_filter(softmax_out, rgb, bilateral=True,
                                                          theta_alpha=self.theta_alpha, theta_beta=self.theta_beta)
            bilateral_out = bilateral_out / bilateral_norm_vals

            # Weighting filter outputs [n_class, w, h]
            message_passing = (tf.matmul(self.spatial_ker_weights,   tf.reshape(spatial_out,   (self.num_class, -1))) +
                               tf.matmul(self.bilateral_ker_weights, tf.reshape(bilateral_out, (self.num_class, -1))))

            # Compatibility transform
            pairwise = tf.matmul(self.compatibility_matrix, message_passing)  # [n_class, w, h]

            # Adding unary potentials
            pairwise = tf.reshape(pairwise, (self.num_class, image_shape[1].value, image_shape[2].value))
            q_values = unaries - pairwise  # [n_class, w, h]

        out_shape = (1, self.num_class, image_shape[1].value, image_shape[2].value)
        return tf.transpose(tf.reshape(q_values, out_shape), perm=(0, 2, 3, 1))  # [1, w, h, n_class]

    @staticmethod
    def get_weights(shape, name):
        return tf.get_variable(name, initializer=tf.truncated_normal(shape))
        # return tf.Variable(tf.truncated_normal(shape), name=name)

    pass
