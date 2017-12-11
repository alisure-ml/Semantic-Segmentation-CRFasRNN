import numpy as np
import tensorflow as tf
from tensorflow.python.ops import gradient_checker
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
custom_module = tf.load_op_library('./cpp/high_dim_filter.so')


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


class HighDimGradTest(object):

    @staticmethod
    def test_high_dim_filter_grad():
        x_shape = [5, 10, 10]

        # Test inputs: unaries and RGB values
        unary_np = np.random.randn(*x_shape).astype(np.float32)
        rgb_np = np.random.randint(low=0, high=256, size=x_shape).astype(np.float32)

        with tf.Session() as sess:
            unary_tf = constant_op.constant(unary_np)
            rgb_tf = constant_op.constant(rgb_np)
            y_tf = custom_module.high_dim_filter(unary_tf, rgb_tf, bilateral=True,
                                                 theta_alpha=1000., theta_beta=1000., theta_gamma=1000.)

            out = gradient_checker.compute_gradient([unary_tf, rgb_tf], [x_shape, x_shape], y_tf, x_shape)

            # We only need to compare gradients w.r.t. unaries
            computed = out[0][0].flatten()
            estimated = out[0][1].flatten()

            mask = (computed != 0)
            computed = computed[mask]
            estimated = estimated[mask]
            difference = computed - estimated

            measure1 = np.mean(difference) / np.mean(computed)
            measure2 = np.max(difference) / np.max(computed)

            print("Gradient check: measure1 = {:.6f}, measure2 = {:.6f}".format(measure1, measure2))
            print(measure1, 1e-3, "Errors found in the gradient computation.")
            print(measure2, 2e-2, "Errors found in the gradient computation.")
            print("Gradient check: success!")

        pass

    pass

if __name__ == "__main__":
    HighDimGradTest.test_high_dim_filter_grad()
