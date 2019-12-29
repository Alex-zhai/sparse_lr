# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2018/5/16 15:13
from __future__ import absolute_import, print_function, division
import tensorflow as tf


def parse_single_line(example_proto):
    features = {
        "ids": tf.VarLenFeature(tf.int64),
        "values": tf.VarLenFeature(tf.float32),
        "label": tf.FixedLenFeature([], tf.float32, default_value=0)
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    return parsed_features["label"], parsed_features["ids"], parsed_features["values"]


def get_batch(file_path, batch_size, training):
    files = tf.data.Dataset.list_files(file_path + 'part*')
    dataset = files.apply(
        tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=1, sloppy=True))
    if training:
        dataset = dataset.repeat()
        dataset = dataset.shuffle(10000)
    dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=parse_single_line, batch_size=batch_size,
                                                          num_parallel_batches=1))
    dataset = dataset.prefetch(batch_size)
    # iterator = dataset.make_one_shot_iterator()
    # batch_labels, batch_ids, batch_values = iterator.get_next()
    # # return batch_labels, {"features": {"ids": batch_ids, "values": batch_values}}
    # return batch_labels, batch_ids, batch_values
    return dataset

# if __name__ == '__main__':
#     file_path = "part-r-00000"
#     batch_labels, batch_ids, batch_values = get_batch(file_path, batch_size=1, training=True)
#     sess = tf.Session()
#     for i in range(1):
#         print(sess.run([batch_labels, batch_ids, batch_values]))
