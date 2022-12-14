import pickle
import random

import tensorflow as tf

import config
import dataloader.list_file as lt
from utils import readPFM
from dataloader.data_loader import DataLoaderKITTI, DataLoaderKITTI_SUBMISSION

import matplotlib.pyplot as plt


def img_loader(path):
    """
    载入图像
    :param path: 路径
    :return: 读取的图像
    """
    return tf.image.decode_image(
        tf.io.read_file(path)
    )


def mean_std_normalize(inputs, width, height):
    """
    对图像进行归一化
    :param width: 图像高度
    :param height: 图像高度
    :param inputs: 输入的图像
    :return: 归一化之后的图像
    """
    inputs = tf.cast(inputs, dtype=tf.float32) / 255.
    # 使用固定值进行归一化
    mean_shift = tf.stack([
        tf.ones([height, width]) * 0.485,
        tf.ones([height, width]) * 0.456,
        tf.ones([height, width]) * 0.406,
    ], axis=-1)
    std_scale = tf.stack([
        tf.ones([height, width]) * 0.229,
        tf.ones([height, width]) * 0.224,
        tf.ones([height, width]) * 0.225,
    ], axis=-1)
    inputs = (inputs - mean_shift) / std_scale
    return inputs


def load_image_disp(left_path, right_path, disp_path, is_training):
    """
    载入图像和视差图 并惊醒augmentation
    :param left_path: 左图路径
    :param right_path: 右图路径
    :param disp_path: 视差图路径
    :param is_training: 是否为训练模式
    :return: 左右图像和视差图
    """
    left_img = img_loader(left_path)
    right_img = img_loader(right_path)
    disp = tf.py_function(readPFM, [disp_path], tf.float32)

    if is_training:
        # 训练时随机裁剪
        th, tw = tf.constant(config.TRAIN_CROP_HEIGHT), tf.constant(config.TRAIN_CROP_WIDTH)
        x1 = tf.random_uniform((), minval=0, maxval=config.IMG_WIDTH - tw, dtype=tf.int32)
        y1 = tf.random_uniform((), minval=0, maxval=config.IMG_HEIGHT - th, dtype=tf.int32)

        left_img = tf.image.crop_to_bounding_box(left_img, y1, x1, th, tw)
        right_img = tf.image.crop_to_bounding_box(right_img, y1, x1, th, tw)
        disp = disp[y1:y1 + th, x1:x1 + tw]
        # 归一化
        left_img = mean_std_normalize(left_img, tw, th)
        right_img = mean_std_normalize(right_img, tw, th)
    else:
        img_size = tf.shape(left_img)
        w, h = img_size[1], img_size[0]
        left_img = tf.image.crop_to_bounding_box(left_img, 0, 0, h, w)
        right_img = tf.image.crop_to_bounding_box(right_img, 0, 0, h, w)
        # 归一化
        left_img = mean_std_normalize(left_img, config.IMG_WIDTH, config.IMG_HEIGHT)
        right_img = mean_std_normalize(right_img, config.IMG_WIDTH, config.IMG_HEIGHT)

    return left_img, right_img, disp


def get_dataset(data_path='dataset/', epoch=10, batch_size=18,
                num_threads=10, shuffle_buffer_size=100, is_training=True, ):
    # 读取文件路径
    all_left_img, all_right_img, all_left_disp, \
    test_left_img, test_right_img, test_left_disp = lt.get_sceneflow_img(data_path)
    dataset_len = len(all_right_img)
    all_left_img = tf.constant(all_left_img)
    all_right_img = tf.constant(all_right_img)
    all_disp = tf.constant(all_left_disp)

    # 载入并处理图像
    # 随机分批处理
    dataset = tf.data.Dataset.from_tensor_slices((all_left_img, all_right_img, all_disp)) \
        .map(lambda left_path, right_path, disp_path:
             load_image_disp(left_path, right_path, disp_path, is_training), num_parallel_calls=num_threads) \
        .shuffle(shuffle_buffer_size) \
        .batch(batch_size) \
        .repeat(epoch) \
        .prefetch(batch_size)

    return dataset, dataset_len


if __name__ == '__main__':
    # dataset, _ = get_dataset(is_training=True)
    # iterator = dataset.make_initializable_iterator()
    # next_element = iterator.get_next()
    # with tf.Session() as sess:
    #     sess.run(iterator.initializer)
    #     for _ in range(10):
    #         X = sess.run(next_element)
    #
    #         plt.imshow(X[0][0])
    #         plt.show()
    #         plt.imshow(X[1][0])
    #         plt.show()
    #         plt.imshow(X[2][0])
    #         plt.show()
    #         print(X[0].shape)

    # # dg = DataLoaderSceneFlow(data_path='./dataset/', batch_size=config.TRAIN_BATCH_SIZE, max_disp=192)
    # dg = DataLoaderKITTI(batch_size=config.TRAIN_BATCH_SIZE, max_disp=192)
    # for step, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(dg.generator(is_training=True)):
    #     plt.imshow(imgL_crop[0])
    #     plt.show()
    #     plt.imshow(imgR_crop[0])
    #     plt.show()
    #     plt.imshow(disp_crop_L[0])
    #     plt.show()
    #
    #     if step > 10: break
    # dg = DataLoaderSceneFlow(data_path='./dataset/', batch_size=config.TRAIN_BATCH_SIZE, max_disp=192)
    # dg = DataLoaderKITTI_SUBMISSION()
    # for step, (imgL, imgR, _) in enumerate(dg.generator()):
    #     plt.imshow(imgL[0])
    #     plt.show()
    #     plt.imshow(imgR[0])
    #     plt.show()
    #
    #     if step > 10: break
    # 预处理数据集
    all_left_img, all_right_img, all_left_disp, \
    test_left_img, test_right_img, test_left_disp = lt.get_sceneflow_img('./dataset/')
    print(all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp)
