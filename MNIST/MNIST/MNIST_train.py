import os
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import MNIST_inference
import MNIST_eval


#NET PARA
BATCH_SIZE = 100
LEARNING_RATE_BASE = 1e-3
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 300000
MOVING_AVERAGE_DECAY = 0.99



def train_cnn(mnist):
    MODEL_SAVE_PATH = 'D:/trainModel/MNIST_cnn/'
    LOG_SAVE_PATH = 'D:/trainModel/logs/cnn'
    MODEL_NAME = 'mnistModel.ckpt'
    with tf.compat.v1.name_scope('input'):
        x = tf.compat.v1.placeholder(tf.compat.v1.float32, [BATCH_SIZE, MNIST_inference.IMAGE_SIZE, MNIST_inference.IMAGE_SIZE, MNIST_inference.NUM_CHANNELS], name='x-input')
        y_ = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, MNIST_inference.OUTPUT_NODE], name='y-input')

    #show input images
    #with tf.compat.v1.name_scope('input_reshape'):
    #    image_shaped_input = tf.compat.v1.reshape(x, [-1, MNIST_inference.IMAGE_SIZE, MNIST_inference.IMAGE_SIZE,  MNIST_inference.NUM_CHANNELS])
    #    tf.compat.v1.summary.image('input', image_shaped_input, 10)

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    y = MNIST_inference.inference_cnn(x, True, regularizer)
    
    global_step = tf.compat.v1.Variable(0, trainable=False)

    with tf.compat.v1.name_scope('moving_avg'):
        variable_averages = tf.compat.v1.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)
        variable_averages_op = variable_averages.apply(tf.compat.v1.trainable_variables())
        #tf.compat.v1.summary.scalar('variable averages', variable_averages)

    with tf.compat.v1.name_scope('loss_function'):
        cross_entropy = tf.compat.v1.nn.sparse_softmax_cross_entropy_with_logits(
            logits=y, labels=tf.compat.v1.argmax(y_, 1))
        cross_entropy_mean = tf.compat.v1.reduce_mean(cross_entropy)
        loss = cross_entropy_mean + tf.compat.v1.add_n(tf.compat.v1.get_collection('losses'))
        tf.compat.v1.summary.scalar('loss', loss)

    with tf.compat.v1.name_scope('train_steps'):
        learning_rate = tf.compat.v1.train.exponential_decay(
            LEARNING_RATE_BASE,
            global_step,
            mnist.train.num_examples / BATCH_SIZE,
            LEARNING_RATE_DECAY)
        train_step = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
        tf.compat.v1.summary.scalar('learning rate', learning_rate)
        with tf.compat.v1.control_dependencies([train_step, variable_averages_op]):
            train_op = tf.compat.v1.no_op(name='train')
    
    merged = tf.compat.v1.summary.merge_all()
    saver = tf.compat.v1.train.Saver()

    with tf.compat.v1.Session() as sess:
        #加载模型
        ckpt = tf.train.get_checkpoint_state('D:/trainModel/MNIST_cnn/')
        if ckpt and ckpt.model_checkpoint_path:      
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            tf.compat.v1.global_variables_initializer().run()
            print('initialized!')
        file_writer = tf.compat.v1.summary.FileWriter(LOG_SAVE_PATH, sess.graph, filename_suffix='cnn')
        print('go on training....')
        for i in range(TRAINING_STEPS):
            run_options = tf.compat.v1.RunOptions(
                trace_level = tf.compat.v1.RunOptions.FULL_TRACE)
            run_metadata = tf.compat.v1.RunMetadata()
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, (BATCH_SIZE, MNIST_inference.IMAGE_SIZE, MNIST_inference.IMAGE_SIZE, MNIST_inference.NUM_CHANNELS))
            summary,  _, loss_value, step = sess.run([merged, train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})
            if i % 1000 == 0:
                print('steps: %d, loss: %g.' % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step)
                print('saving metadata....')
                file_writer.add_run_metadata(run_metadata, 'step:%d' % step)
                file_writer.add_summary(summary, step)
                file_writer.flush()
                print('saved!')
        file_writer.close()
        print('file_writer closed!')

def train_fp(mnist):
    MODEL_SAVE_PATH = 'D:/trainModel/MNIST/'
    LOG_SAVE_PATH = 'D:/trainModel/logs/fp'
    MODEL_NAME = 'mnistModel.ckpt'
    with tf.compat.v1.name_scope('input'):
        x = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, MNIST_inference.INPUT_NODE], name='x-input')
        y_ = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, MNIST_inference.OUTPUT_NODE], name='y-input')

    #show input images
    with tf.compat.v1.name_scope('input_reshape'):
        image_shaped_input = tf.compat.v1.reshape(x, [-1, 28 ,28, 1])
        tf.compat.v1.summary.image('input', image_shaped_input, 10)

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    y = MNIST_inference.inference_fp(x, regularizer)
    
    global_step = tf.compat.v1.Variable(0, trainable=False)

    with tf.compat.v1.name_scope('moving_avg'):
        variable_averages = tf.compat.v1.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)
        variable_averages_op = variable_averages.apply(tf.compat.v1.trainable_variables())
        #tf.compat.v1.summary.scalar('variable averages', variable_averages)

    with tf.compat.v1.name_scope('loss_function'):
        cross_entropy = tf.compat.v1.nn.sparse_softmax_cross_entropy_with_logits(
            logits=y, labels=tf.compat.v1.argmax(y_, 1))
        cross_entropy_mean = tf.compat.v1.reduce_mean(cross_entropy)
        loss = cross_entropy_mean + tf.compat.v1.add_n(tf.compat.v1.get_collection('losses'))
        tf.compat.v1.summary.scalar('loss', loss)

    with tf.compat.v1.name_scope('train_steps'):
        learning_rate = tf.compat.v1.train.exponential_decay(
            LEARNING_RATE_BASE,
            global_step,
            mnist.train.num_examples / BATCH_SIZE,
            LEARNING_RATE_DECAY)
        train_step = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        tf.compat.v1.summary.scalar('learning rate', learning_rate)
        with tf.compat.v1.control_dependencies([train_step, variable_averages_op]):
            train_op = tf.compat.v1.no_op(name='train')
    
    merged = tf.compat.v1.summary.merge_all()
    saver = tf.compat.v1.train.Saver()

    with tf.compat.v1.Session() as sess:
        tf.compat.v1.global_variables_initializer().run()
        print('initialized!')
        file_writer = tf.compat.v1.summary.FileWriter(LOG_SAVE_PATH, sess.graph, filename_suffix='fp')
        print('go on training....')
        for i in range(TRAINING_STEPS):
            run_options = tf.compat.v1.RunOptions(
                trace_level = tf.compat.v1.RunOptions.FULL_TRACE)
            run_metadata = tf.compat.v1.RunMetadata()
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            summary,  _, loss_value, step = sess.run([merged, train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print('steps: %d, loss: %g.' % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step)
                print('saving metadata....')
                file_writer.add_run_metadata(run_metadata, 'step:%d' % step)
                file_writer.add_summary(summary, step)
                print('saved!')
        file_writer.close()
        print('file_writer closed!')

def main(argv = None):
    mnist = input_data.read_data_sets("MNIST_data", one_hot= True)
    print("----------FP----------")
    #train_fp(mnist)
    print("----------cnn----------")
    train_cnn(mnist)

if __name__ == '__main__':
    main()
