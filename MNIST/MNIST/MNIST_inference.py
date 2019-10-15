import tensorflow as tf

#para
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500
IMAGE_SIZE = 28
NUM_CHANNELS = 1
CONV1_DEEP = 32
CONV1_SIZE = 5
CONV2_DEEP = 64
CONV2_SIZE = 5
FC_SIZE = 512

#summaries genarate
def variable_summaries(var, name):
    with tf.name_scope('summaries'):
        tf.summary.histogram(name, var)
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/'+ name, mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev/'+ name, stddev)
#for FC layer weights generate
def get_weight_variable(shape, regularizer):
    weights = tf.get_variable('weights', shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights


def inference_fp(input, regularizer):
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable('biases', [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        variable_summaries(weights, 'fc1/weights')
        variable_summaries(biases, 'fc1/biases')
        preactivation = tf.matmul(input, weights) + biases
        tf.summary.histogram('fc1/pre_activations', preactivation)
        layer1 = tf.nn.relu(preactivation)
        tf.summary.histogram('fc1/activations', layer1)

    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable('biases', [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        variable_summaries(weights, 'fc2/weights')
        variable_summaries(biases, 'fc2/biases')
        layer2 = tf.matmul(layer1, weights) + biases
    return layer2

def inference_cnn(input, train, regularizer):
    #input:28*28*1; filter:weights 5, deep 32, strides 1 ,same
    with tf.variable_scope('conv1'):
        conv1_weights = tf.get_variable('weight', [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable('biases', [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input, conv1_weights, [1, 1, 1, 1], 'SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.variable_scope('pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
    #input:14*14*32; filter:weights 5, deep 64, strides 1 ,same
    with tf.variable_scope('conv2'):
        conv2_weights = tf.get_variable('weight', [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable('biases', [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.variable_scope('pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    #trasform
    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    with tf.variable_scope('fc1'):
       # fc1_weights = get_weight_variable([nodes, FC_SIZE], regularizer)
        fc1_weights = tf.get_variable('weights', [nodes, FC_SIZE], initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable('biases', [FC_SIZE], initializer=tf.constant_initializer(0.1))
        variable_summaries(fc1_weights, 'fc1/weights')
        variable_summaries(fc1_biases, 'fc1/biases')
        preactivation = tf.matmul(reshaped, fc1_weights) + fc1_biases
        tf.summary.histogram('fc1/pre_activations', preactivation)
        fc1 = tf.nn.relu(preactivation)
        if train: fc1 = tf.nn.dropout(fc1, 0.5)
        tf.summary.histogram('fc1/activations', fc1)


    with tf.variable_scope('fc2'):
        #fc2_weights = get_weight_variable([FC_SIZE, OUTPUT_NODE], regularizer)
        fc2_weights = tf.get_variable('weights', [FC_SIZE, OUTPUT_NODE], initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable('biases', [OUTPUT_NODE], initializer=tf.constant_initializer(0.1))
        variable_summaries(fc2_weights, 'fc2/weights')
        variable_summaries(fc2_biases, 'fc2/biases')
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases
        
    return logit