# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2018/5/17 14:32

from __future__ import print_function, division, absolute_import

import datetime
import os

import tensorflow as tf

# tf.enable_eager_execution()


tf.app.flags.DEFINE_string('checkpoint_path', default="sparse_lr_model", help="the path of checkpoint")
tf.app.flags.DEFINE_integer("train_batch_size", 64, "The batch size of training")
tf.app.flags.DEFINE_integer("validation_batch_size", 64, "The batch size of training")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "The learning rate")
tf.app.flags.DEFINE_string("output_path", "sparse_tensorboard/", "The path of tensorboard event files")
tf.app.flags.DEFINE_integer("steps_to_validate", 1, "Steps to validate and print state")
tf.app.flags.DEFINE_boolean('training', default=True, help="do training and valuation ")
tf.app.flags.DEFINE_boolean('testing', default=False, help="do testing")
tf.app.flags.DEFINE_boolean('local_inference', default=False, help="do local inference")
tf.app.flags.DEFINE_boolean('serving', default=False,
                            help="whether to export a tf serving savedModel for doing large-scale inference")

FLAGS = tf.app.flags.FLAGS

# define model variables
weights = tf.get_variable("weights", shape=[1220331, 2], initializer=tf.random_normal_initializer())
biases = tf.get_variable("biases", shape=[2], initializer=tf.random_normal_initializer())


def restore_session_from_checkpoint(sess, saver, checkpoint):
    if checkpoint:
        print("Restore session from checkpoint: {}".format(checkpoint))
        saver.restore(sess, checkpoint)
        return True
    else:
        return False


def parse_single_line(example_proto):
    features = {
        "ids": tf.VarLenFeature(tf.int64),
        "values": tf.VarLenFeature(tf.float32),
        "label": tf.FixedLenFeature([], tf.float32, default_value=0)
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    return parsed_features["label"], parsed_features["ids"], parsed_features["values"]


def sparse_lr_model(sparse_ids, sparse_values):
    return tf.nn.bias_add(tf.nn.embedding_lookup_sparse(weights, sparse_ids, sparse_values, combiner="sum"), biases)


def train_and_eval(train_file_path, eval_file_path):
    if not os.path.exists(FLAGS.checkpoint_path):
        os.makedirs(FLAGS.checkpoint_path)
    checkpoint_file_path = FLAGS.checkpoint_path + "/checkpoint.ckpt"
    latest_checkpoint_file_path = tf.train.latest_checkpoint(
        FLAGS.checkpoint_path)

    if not os.path.exists(FLAGS.output_path):
        os.makedirs(FLAGS.output_path)

    # step1: Construct the dataset op
    train_filename_list = [train_file_path + filename for filename in os.listdir(train_file_path)]
    print(train_filename_list)
    train_filename_placeholder = tf.placeholder(tf.string, shape=[None])
    train_dataset = tf.data.TFRecordDataset(train_filename_placeholder)
    train_dataset = train_dataset.map(parse_single_line).repeat().shuffle(1000).batch(FLAGS.train_batch_size).prefetch(
        FLAGS.train_batch_size)
    train_dataset_iterator = train_dataset.make_initializable_iterator()
    batch_labels, batch_ids, batch_values = train_dataset_iterator.get_next()

    print(batch_labels, batch_ids)

    validation_filename_list = [eval_file_path + filename for filename in os.listdir(eval_file_path)]
    validation_filename_placeholder = tf.placeholder(tf.string, shape=[None])
    validation_dataset = tf.data.TFRecordDataset(validation_filename_placeholder)
    validation_dataset = validation_dataset.map(parse_single_line).batch(FLAGS.validation_batch_size).prefetch(
        FLAGS.validation_batch_size)
    validation_dataset_iterator = validation_dataset.make_initializable_iterator()
    validation_labels, validation_ids, validation_values = validation_dataset_iterator.get_next()

    # step2: define the model op
    logits = sparse_lr_model(batch_ids, batch_values)
    batch_labels = tf.to_int64(batch_labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=batch_labels)
    loss = tf.reduce_mean(cross_entropy, name="loss")
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)
    tf.get_variable_scope().reuse_variables()

    # Define accuracy op for train train_data
    train_accuracy_logits = sparse_lr_model(batch_ids, batch_values)
    train_softmax = tf.nn.softmax(train_accuracy_logits)
    train_correct_prediction = tf.equal(
        tf.argmax(train_softmax, 1), batch_labels)
    train_accuracy = tf.reduce_mean(
        tf.cast(train_correct_prediction, tf.float32))

    # Define accuracy op for validate train_data
    validate_accuracy_logits = sparse_lr_model(validation_ids, validation_values)
    validate_softmax = tf.nn.softmax(validate_accuracy_logits)
    validate_batch_labels = tf.to_int64(validation_labels)
    validate_correct_prediction = tf.equal(
        tf.argmax(validate_softmax, 1), validate_batch_labels)
    validate_accuracy = tf.reduce_mean(
        tf.cast(validate_correct_prediction, tf.float32))

    # saver and summary
    saver = tf.train.Saver()
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("train_accuracy", train_accuracy)
    tf.summary.scalar("validate_accuracy", validate_accuracy)
    summary_op = tf.summary.merge_all()

    init_op = [
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    ]

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(FLAGS.output_path, sess.graph)
        sess.run(init_op)
        sess.run(
            train_dataset_iterator.initializer,
            feed_dict={train_filename_placeholder: train_filename_list})
        sess.run(
            validation_dataset_iterator.initializer,
            feed_dict={validation_filename_placeholder: validation_filename_list})

        restore_session_from_checkpoint(sess, saver, latest_checkpoint_file_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        start_time = datetime.datetime.now()

        try:
            while not coord.should_stop():
                _, step = sess.run([train_op, global_step])

                if step % FLAGS.steps_to_validate == 0:
                    loss_value, train_accuracy_value, validate_accuracy_value, summary_value \
                        = sess.run([loss, train_accuracy, validate_accuracy, summary_op])
                    end_time = datetime.datetime.now()
                    print(
                        "[{}] Step: {}, loss: {}, train_acc: {}, valid_acc: {}".
                            format(end_time - start_time, step, loss_value,
                                   train_accuracy_value, validate_accuracy_value))
                    writer.add_summary(summary_value, step)
                    saver.save(sess, checkpoint_file_path, global_step=step)
                    start_time = end_time

        except tf.errors.OutOfRangeError:
            print("Finish training ")
            exit(0)

        finally:
            coord.request_stop()
        coord.join(threads)


def main(_):
    if FLAGS.training:
        train_and_eval("train_data/", "val_data/")


if __name__ == "__main__":
    tf.app.run()
