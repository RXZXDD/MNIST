import tensorflow as tf
import MNIST_inference
import MNIST_train
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import PIL.ImageOps

def modelEval(mnist):
    x = tf.placeholder(tf.float32, [None, MNIST_inference.INPUT_NODE],'x-input')
    y_ = tf.placeholder(tf.float32, [None, MNIST_inference.OUTPUT_NODE], 'y-hat')

    #reshaped_xs = np.reshape(xs, (BATCH_SIZE, MNIST_inference.IMAGE_SIZE, MNIST_inference.IMAGE_SIZE, MNIST_inference.NUM_CHANNELS))
    validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
    
    y = MNIST_inference.inference_fp(x, None)

    prediction = tf.argmax(y, 1)

    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_,1), prediction), tf.float32))

    variable_averages = tf.train.ExponentialMovingAverage(MNIST_train.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state('D:/trainModel/MNIST/')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            acc_score = sess.run(acc, feed_dict=validate_feed)
            print("train step:{0} acc: {1}".format(global_step, acc_score))
        else:
            print("404 No checkpoint found")

def modelEval_cnn(mnist):
    #对训练模型进行评价,输出在MNIST测试集上的精确度


    x = tf.placeholder(tf.float32, [mnist.test.num_examples, MNIST_inference.IMAGE_SIZE, MNIST_inference.IMAGE_SIZE, MNIST_inference.NUM_CHANNELS],'x-input')
    y_ = tf.placeholder(tf.float32, [None, MNIST_inference.OUTPUT_NODE], 'y-hat')

    #xs, ys = mnist.validation
    reshaped_xs = np.reshape(mnist.test.images, (mnist.test.num_examples, MNIST_inference.IMAGE_SIZE, MNIST_inference.IMAGE_SIZE, MNIST_inference.NUM_CHANNELS))
    validate_feed = {x: reshaped_xs, y_: mnist.test.labels}
    
    y = MNIST_inference.inference_cnn(x, False, None)

    prediction = tf.argmax(y, 1)

    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_,1), prediction), tf.float32))

    variable_averages = tf.train.ExponentialMovingAverage(MNIST_train.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver()

    

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state('D:/trainModel/MNIST_cnn/')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            acc_score = sess.run(acc, feed_dict=validate_feed)
            print("train step:{0} acc: {1}".format(global_step, acc_score))
        else:
            print("404 No checkpoint found")

def recognition(inputPath):
    #输入图片路径,并对图片进行数字识别


    input = Image.open(inputPath, "r")
    input = PIL.ImageOps.invert(input)
    inputImg = input.resize((MNIST_inference.IMAGE_SIZE, MNIST_inference.IMAGE_SIZE),Image.ANTIALIAS)
    
    inputImgInput = inputImg.convert('L')
    inputImg.show()
    

    x = tf.placeholder(tf.float32, [1, MNIST_inference.IMAGE_SIZE, MNIST_inference.IMAGE_SIZE, MNIST_inference.NUM_CHANNELS],'x-input')
    y_ = tf.placeholder(tf.float32, [None, MNIST_inference.OUTPUT_NODE], 'y-hat')

    reshaped_xs = np.reshape(inputImgInput, (1, MNIST_inference.IMAGE_SIZE, MNIST_inference.IMAGE_SIZE, MNIST_inference.NUM_CHANNELS))
    
    y = MNIST_inference.inference_cnn(x, False, None)

    prediction = tf.argmax(y, 1)

    variable_averages = tf.train.ExponentialMovingAverage(MNIST_train.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver()

    

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state('D:/trainModel/MNIST_cnn/')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            _y, prediction_ = sess.run([y, prediction], feed_dict={x: reshaped_xs})
            print("y[]:{0} prediction: {1}".format(_y, prediction_))
            #prediction_, ipnut_label = sess.run(prediction, tf.argmax(y_,1), feed_dict={})
        else:
            print("404 No checkpoint found")

def main(argv = None):
    mnist = input_data.read_data_sets("MNIST_data", one_hot= True)
    #modelEval_cnn(mnist)
    recognition("D:\Workspaces\MNIST\img\\2-1.png")

if __name__ == '__main__':
    main()