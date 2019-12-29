# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2018/5/15 10:32

from __future__ import absolute_import, print_function, division
import os
import tensorflow as tf
import time


def convent_tfrecord(input_filename, output_filename):
    print("Start to convert {} to {}".format(input_filename, output_filename))
    writer = tf.python_io.TFRecordWriter(output_filename)

    lines = open(input_filename, "r").readlines()
    for line in lines:
        data = line.split(" ")
        label = int(data[0])
        ids = []
        values = []
        for feat in data[1:]:
            id, value = feat.split(":")
            ids.append(int(id))
            values.append(float(value))
        example = tf.train.Example(features=tf.train.Features(
            feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                "ids": tf.train.Feature(int64_list=tf.train.Int64List(value=ids)),
                "values": tf.train.Feature(float_list=tf.train.FloatList(value=values))
            }
        ))
        writer.write(example.SerializeToString())
    writer.close()
    print("Successfully convert {} to {}".format(input_filename, output_filename))


def gen_all_tfrecords(file_path):
    for filename in os.listdir(file_path):
        file = file_path + filename
        convent_tfrecord(file, filename + ".tfrecords")


def main():
    file_path = "small_gender_train"
    gen_all_tfrecords(file_path)


if __name__ == '__main__':
    start = time.time()
    main()
    print("consumed time is ", time.time() - start)
