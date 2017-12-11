from model import CRFasRNN
import util
import tensorflow as tf


def main(input_file="data/image.jpg", output_file="result/crf_result.png"):
    # 构造模型
    crf_as_rnn_model = CRFasRNN(500, 500, num_class=8)
    input_image = crf_as_rnn_model.img_input
    output_op = crf_as_rnn_model.build_net(input_image)

    # 读数据、减均值、pad大小到500, 返回处理后的数据和原始大小
    img_data, img_h, img_w = util.get_preprocessed_image(input_file)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        probs = sess.run(output_op, feed_dict={input_image: img_data})

        # 从概率到着色图片
        segmentation = util.get_label_image(probs[0], img_h, img_w)
        segmentation.save(output_file)

    pass


if __name__ == "__main__":
    main()
