import numpy as np
import tensorflow as tf
import time
import math
import helper

train_file = "data/dbpedia.train"
test_file = "data/dbpedia.test"

max_window_size = 1000
batch_size = 500

label_to_int,vocab_to_int = helper.build_vocab_dict(train_file)
print (label_to_int)
print (vocab_to_int)

train_data = helper.read_data_into_cache(train_file)
test_data = helper.read_data_into_cache(test_file)

def get_batches(start_pos, batch_size, train_data):
    batch = train_data[start_pos:start_pos + batch_size]
    x = np.zeros((batch_size, max_window_size))
    mask = np.zeros((batch_size, max_window_size))
    y = []
    word_num = np.zeros((batch_size))
    line_no = 0
    for line in batch:
        line = line.strip().split(' ')
        if '__label__1' in line[0]:
            y.append(1)
        else:
            y.append(0)
        col_no = 0
        for i in line[1:]:
            if i in vocab_to_int:
                x[line_no][col_no] = vocab_to_int[i]
                mask[line_no][col_no] = 1
                col_no += 1
            if col_no >= max_window_size:
                break
        word_num[line_no] = col_no
        line_no += 1
    #y = np.array(y)
    y = np.array(y).reshape(batch_size, 1)
    #y = tf.concat(1, [1 - y, y])
    #y_test = tf.concat(1, [1 - y_test, y_test])
    batch_mask = mask.reshape(batch_size, max_window_size, 1)
    word_number = word_num.reshape(batch_size, 1)
    return x,y,batch_mask,word_number

def get_inputs():
    inputs = tf.placeholder(tf.int32, shape=[None, max_window_size])
    targets = tf.placeholder(tf.int32, [None, 1])
    vocab_num = tf.placeholder(tf.float32, shape=[None, 1])
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    return inputs, targets, learning_rate, vocab_num

vocab_size = len(vocab_to_int)+1
embed_dim = 128
def get_embed(input_data, vocab_size, embed_dim):
    emb_mask = tf.placeholder(tf.float32, shape=[None, max_window_size, 1])
    embedding = tf.Variable(tf.random_uniform((vocab_size, embed_dim), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, input_data)
    project_embedding = tf.div(tf.reduce_sum(tf.multiply(embed, emb_mask), 1),vocab_size)
    return embed,project_embedding,emb_mask


def build_relu_layer(x,width):
    weights = tf.Variable(tf.random_normal([embed_dim, width]))
    biases = tf.Variable(tf.random_normal([width]))
    layer = tf.add(tf.matmul(x, weights), biases)
    layer = tf.nn.relu(layer)
    return layer


def build_pred_layer(x,input_width,class_number):
    weights = tf.Variable(tf.random_normal([input_width, class_number]))
    biases = tf.Variable(tf.random_normal([class_number]))
    layer = tf.add(tf.matmul(x, weights), biases)
    return layer

relu_layer_width = 128
class_number = 1
train_graph = tf.Graph()
with train_graph.as_default():
    inputs, targets, learning_rate,vocab_num = get_inputs()
    embed,project_embedding,emb_mask = get_embed(inputs, vocab_size, embed_dim)
    # Construct model
    relu_output = build_relu_layer(project_embedding,relu_layer_width)
    pred = build_pred_layer(relu_output,relu_layer_width,class_number)
    print(pred)
    print(targets)

    predictions = tf.contrib.layers.fully_connected(pred, 1, activation_fn=tf.sigmoid)
    cost = tf.losses.mean_squared_error(targets, predictions)

    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, targets))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(targets, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Number of Epochs
num_epochs = 1
batch_size = 300
learning_rate_value = 0.01
show_every_n_batches = 10

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    # Training cycle
    start_time = time.time()
    total_batch = int(len(vocab_to_int) / batch_size)
    print("total_batch of training data: ", total_batch)
    for epoch in range(num_epochs):
        avg_cost = 0.
        for i in range(total_batch):
            x, y, batch_mask, word_number = get_batches(i * batch_size, batch_size, train_data)
            feed = {inputs: x, targets: y,
                    emb_mask: batch_mask, vocab_num: word_number,
                    learning_rate:learning_rate_value}
            loss, _ = sess.run([cost, optimizer], feed_dict=feed)

            if i%2==0:
                print("Epoch: {}/{}".format(epoch, num_epochs),
                      "Iteration: {}".format(i),
                      "Train loss: {:.3f}".format(loss))

    test_acc = []
    test_pred = []
    total_batch = int(len(test_data) / batch_size)
    final_accuracy = 0
    for i in range(total_batch):
        x, y, batch_mask, word_number = get_batches(i*batch_size, batch_size, test_data)
        feed = {inputs: x, targets: y,
                emb_mask: batch_mask, vocab_num: word_number,
                learning_rate:learning_rate_value}
        batch_acc, batch_test_pred = sess.run([accuracy, pred], feed_dict=feed)
        test_acc.append(batch_acc)
        test_pred.append(batch_test_pred)
    print("Test accuracy: {:.3f}".format(np.mean(test_acc)))
    print("Final Accuracy: ", final_accuracy * 1.0 / total_batch)

