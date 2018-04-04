#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by C.L.Wang
# 生成Android集成TF的测试模型

import tensorflow as tf
from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib

MODEL_FOLDER = "./data/android/"
MODEL_NAME = 'tfdroid'


def gnr_graph_checkpoint():
    """
    生成简单的图和checkpoint
    :return: 
    """
    I = tf.placeholder(tf.float32, shape=[None, 3], name='I')  # input
    W = tf.Variable(tf.zeros(shape=[3, 2]), dtype=tf.float32, name='W')  # weights
    b = tf.Variable(tf.zeros(shape=[2]), dtype=tf.float32, name='b')  # biases
    O = tf.nn.relu(tf.matmul(I, W) + b, name='O')  # activation / output

    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)

        tf.train.write_graph(sess.graph_def, MODEL_FOLDER, 'tfdroid.pbtxt')  # 存储TensorFlow的图

        # 训练数据，本例直接赋值
        sess.run(tf.assign(W, [[1, 2], [4, 5], [7, 8]]))
        sess.run(tf.assign(b, [1, 1]))

        # 存储checkpoint文件，即参数信息
        saver.save(sess, MODEL_FOLDER + 'tfdroid.ckpt')


def gnr_freeze_graph(input_graph, input_saver, input_binary, input_checkpoint,
                     output_node_names, output_graph, clear_devices):
    """
    将输入图与参数结合在一起

    :param input_graph: 输入图
    :param input_saver: Saver解析器
    :param input_binary: 输入图的格式，false是文本，true是二进制
    :param input_checkpoint: checkpoint，检查点文件

    :param output_node_names: 输出节点名称
    :param output_graph: 保存输出文件
    :param clear_devices: 清除训练设备
    :return: NULL
    """
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"

    freeze_graph.freeze_graph(
        input_graph=input_graph,  # 输入图
        input_saver=input_saver,  # Saver解析器
        input_binary=input_binary,  # 输入图的格式，false是文本，true是二进制
        input_checkpoint=input_checkpoint,  # checkpoint，检查点文件
        output_node_names=output_node_names,  # 输出节点名称
        restore_op_name=restore_op_name,  # 从模型恢复节点的名字
        filename_tensor_name=filename_tensor_name,  # tensor名称
        output_graph=output_graph,  # 保存输出文件
        clear_devices=clear_devices,  # 清除训练设备
        initializer_nodes="")  # 初始化节点


def gnr_optimize_graph(graph_path, optimized_graph_path):
    """
    优化图
    :param graph_path: 原始图
    :param optimized_graph_path: 优化的图
    :return: NULL
    """
    input_graph_def = tf.GraphDef()  # 读取原始图
    with tf.gfile.Open(graph_path, "r") as f:
        data = f.read()
        input_graph_def.ParseFromString(data)

    # 设置输入输出节点，剪切分支，大约节省1/4
    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def,
        ["I"],  # an array of the input node(s)
        ["O"],  # an array of output nodes
        tf.float32.as_datatype_enum)

    # 存储优化的图
    f = tf.gfile.FastGFile(optimized_graph_path, "w")
    f.write(output_graph_def.SerializeToString())


if __name__ == "__main__":
    gnr_graph_checkpoint()  # 生成图和参数

    input_graph_path = MODEL_FOLDER + MODEL_NAME + '.pbtxt'  # 输入图
    checkpoint_path = MODEL_FOLDER + MODEL_NAME + '.ckpt'  # 输入参数
    output_path = MODEL_FOLDER + 'frozen_' + MODEL_NAME + '.pb'  # Freeze模型

    # 生成模型
    gnr_freeze_graph(input_graph=input_graph_path, input_saver="",
                     input_binary=False, input_checkpoint=checkpoint_path,
                     output_node_names="O", output_graph=output_path, clear_devices=True)

    optimized_output_graph = MODEL_FOLDER + 'optimized_' + MODEL_NAME + '.pb'

    # 生成优化模型
    gnr_optimize_graph(output_path, optimized_output_graph)
