# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2018/5/16 18:47

from __future__ import print_function, division, absolute_import

import shutil

import tensorflow as tf

tf.app.flags.DEFINE_string("train_data_path", "small_gender_lr_train/", "The glob pattern of train TFRecords files")
tf.app.flags.DEFINE_string("val_data_path", "small_gender_lr_test/", "The glob pattern of eval TFRecords files")
tf.app.flags.DEFINE_string('model_path', default="sparse_lr_model", help="the path of saved model")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "The learning rate")
tf.app.flags.DEFINE_boolean('training', default=True, help="do training and valuation")
tf.app.flags.DEFINE_boolean('testing', default=False, help="do testing")

FLAGS = tf.app.flags.FLAGS


def parse_single_line(example_proto):
    features = {
        "ids": tf.VarLenFeature(tf.int64),
        "values": tf.VarLenFeature(tf.float32),
        "label": tf.FixedLenFeature([], tf.float32, default_value=0)
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    features_dict = {"ids": parsed_features["ids"], "values": parsed_features["values"]}
    return features_dict, parsed_features["label"]


def input_fn(file_path, batch_size=128, training=True):
    files = tf.data.Dataset.list_files(file_path + 'part*')
    dataset = files.apply(
        tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=1, sloppy=True))
    if training:
        dataset = dataset.repeat()
        dataset = dataset.shuffle(10000)
    dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=parse_single_line, batch_size=batch_size,
                                                          num_parallel_batches=1))
    dataset = dataset.prefetch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels


def sparse_lr_model_fn(features, labels, mode, params):
    weights = tf.get_variable("weights", shape=[1220331, 2], initializer=tf.random_normal_initializer())
    biases = tf.get_variable("biases", shape=[2], initializer=tf.random_normal_initializer())
    logits = tf.nn.bias_add(tf.nn.embedding_lookup_sparse(weights, features["ids"], features["values"], combiner="sum"),
                            biases)
    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs={
            'predict': tf.estimator.export.PredictOutput(predictions)
        })
    loss = tf.losses.sparse_softmax_cross_entropy(labels=tf.cast(labels, tf.int64), logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_global_step()
        optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])
        # if params["use_multi_gpus"]:
        #     optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)
        train_op = optimizer.minimize(loss=loss, global_step=global_step)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def train_and_eval(train_file_path, val_file_path, save_model_path):
    shutil.rmtree(save_model_path, ignore_errors=True)
    model_function = sparse_lr_model_fn
    lr_model = tf.estimator.Estimator(
        model_fn=model_function, model_dir=save_model_path, params={
            'learning_rate': FLAGS.learning_rate,
        }
    )
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(train_file_path), max_steps=1000)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(val_file_path, training=False))
    tf.estimator.train_and_evaluate(estimator=lr_model, train_spec=train_spec, eval_spec=eval_spec)


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    if FLAGS.training:
        train_and_eval(FLAGS.train_data_path, FLAGS.val_data_path, FLAGS.model_path)


if __name__ == '__main__':
    tf.app.run()
