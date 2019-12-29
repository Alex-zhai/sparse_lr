# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2018/5/16 21:35

from __future__ import print_function, division, absolute_import

import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

import gen_batch_data

learning_rate = 0.001
batch_size = 32
num_steps = 3000
dis_step = 10

weights = tf.get_variable("weights", shape=[1220331, 2], initializer=tf.random_normal_initializer())
biases = tf.get_variable("biases", shape=[2], initializer=tf.random_normal_initializer())


# def train_input_fn(file_path, batch_size=128, training=True):
#     return get_batch(file_path, batch_size, training)
#
# def eval_input_fn(file_path, batch_size=128, training=False):
#     return get_batch(file_path, batch_size, training)


def sparse_lr_model(sparse_ids, sparse_values):
    return tf.nn.bias_add(tf.nn.embedding_lookup_sparse(weights, sparse_ids, sparse_values, combiner="sum"), biases)


def loss_fn(inference_fn, sparse_ids, sparse_values, labels):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=inference_fn(sparse_ids, sparse_values), labels=labels))


def accuracy_fn(inference_fn, sparse_ids, sparse_values, labels):
    pred = tf.nn.softmax(inference_fn(sparse_ids, sparse_values))
    correct_pred = tf.equal(tf.argmax(pred, 1), labels)
    return tf.reduce_mean(tf.cast(correct_pred, tf.float32))


def train_and_eval(train_file_path, eval_file_path):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    grad = tfe.implicit_gradients(loss_fn)
    dataset = gen_batch_data.get_batch(train_file_path, batch_size=batch_size, training=True)
    dataset_iter = tfe.Iterator(dataset)
    average_loss = 0.
    average_acc = 0.
    for step in range(num_steps):
        try:
            d = dataset_iter.next()
        except StopIteration:
            dataset_iter = tfe.Iterator(dataset)
            d = dataset_iter.next()
        label_batch = tf.cast(d[0], tf.int64)
        ids_batch = d[1]
        value_batch = d[2]
        # Compute the batch loss
        batch_loss = loss_fn(sparse_lr_model, ids_batch, value_batch, label_batch)
        average_loss += batch_loss
        # Compute the batch accuracy
        batch_accuracy = accuracy_fn(sparse_lr_model, ids_batch, value_batch, label_batch)
        average_acc += batch_accuracy

        if step == 0:
            # Display the initial cost, before optimizing
            print("Initial loss= {:.9f}".format(average_loss))

        # Update the variables following gradients info
        optimizer.apply_gradients(grad(sparse_lr_model, ids_batch, value_batch, label_batch))

        # Display info
        if (step + 1) % dis_step == 0 or step == 0:
            if step > 0:
                average_loss /= dis_step
                average_acc /= dis_step
            print("Step:", '%04d' % (step + 1), " loss=",
                  "{:.9f}".format(average_loss), " accuracy=",
                  "{:.4f}".format(average_acc))
            average_loss = 0.
            average_acc = 0.
            # dataset_test = get_batch(eval_file_path, batch_size=batch_size, training=False)


if __name__ == '__main__':
    train_and_eval("train_data/", "train_data/")
