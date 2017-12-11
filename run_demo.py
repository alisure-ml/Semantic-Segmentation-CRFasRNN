from model import CRFasRNN
import util
import tensorflow as tf


def main(input_file="data/image.jpg", output_file="result/labels_new_crf_1.png"):
    crf_as_rnn = CRFasRNN(500, 500, num_class=4)
    # 构造模型
    img_input = crf_as_rnn.img_input
    output = crf_as_rnn.build_net(img_input)

    # 读数据、减均值、pad大小到500, 返回处理后的数据和原始大小
    img_data, img_h, img_w = util.get_preprocessed_image(input_file)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        probs = sess.run(output, feed_dict={img_input: img_data})

        # 从概率到着色图片
        segmentation = util.get_label_image(probs[0], img_h, img_w)
        segmentation.save(output_file)

    pass


if __name__ == "__main__":
    main()
