import os
import random
from PIL import Image, ImageEnhance
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
#
# batch_x, batch_y = mnist.train.next_batch(1)
#
# print(batch_x,batch_y)

def renameData(dir):
    index = 0
    for file in os.listdir(dir):
        if "virus" in file:
            rename_letter = 'v'
        elif "bacteria" in file:
            rename_letter = 'b'
        else:
            rename_letter = 'n'
        dst = rename_letter + str(index) + ".jpg"
        src = dir + file
        dst = dir + dst

        os.rename(src, dst)
        index += 1


def createShuffledListFiles(dir):
    files = []
    new_list = []
    dir_name = dir
    if os.path.isdir(dir_name):
        for file_name in os.listdir(dir_name):
            files.append(file_name)
    else:
        print("Directory", dir_name, "does not exist")
        return
# shuffle list
    random.shuffle(files)
    for item in files:
        new_list.append(os.path.join(dir_name, item))
    return new_list


def img_to_vector(image):
    img = Image.open(image).convert('L')
    pixel_array = np.array(img)
    image_vector = []
    for row in pixel_array:
        for pixel in row:
            image_vector.append((pixel/255))
    return image_vector


def convert_multiple_img_to_vec(list):
    vector_list = []
    for img in list:
        vector_list.append(img_to_vector(img))
    return vector_list


def scalar_to_one_hot(label, num_classes):
    one_hot_vector = [0 for j in range(num_classes)]
    for num in range(num_classes):
        if num == label:
            one_hot_vector[num] = 1
        else:
            one_hot_vector[num] = 0
    return one_hot_vector


def get_label_from_file(fileName, index_of_label):
    letter_label = fileName[index_of_label]
    if letter_label == 'n':
        return 0
    elif letter_label == 'b' or letter_label == 'v':
        return 1
    # elif letter_label == 'n':
    #     return 2



def get_batches(size, list_of_input, list_of_labels):
    batch_x = []
    batch_y = []
    for i in range(size):
        x = random.choice(list_of_input)
        batch_x.append(x)
        batch_y.append(list_of_labels[list_of_input.index(x)])
    return (batch_x, batch_y)


# Hyper-parameters

learning_rate = 0.01
epochs = 600
batch_size = 350
display_step = 100
n_hidden_1 = 600
n_hidden_2 = 600
input_size = 2500
num_classes = 2
# keep_prob = 1  #0.85 during training

# tf Graph input

X = tf.placeholder("float", [None, input_size])
Y = tf.placeholder("float", [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

# init weights and biases

weights = {
    'w1': tf.Variable(tf.random_normal([input_size, n_hidden_1])),
    'w2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# create model

def neural_net(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['w1']), biases['b1']))
    drop_out1 = tf.nn.dropout(layer_1, keep_prob)
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(drop_out1, weights['w2']), biases['b2']))
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


logits = neural_net(X)

# Define error func and optimizer

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# evaluate model
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


init = tf.global_variables_initializer()

saver = tf.train.Saver()


# #
# test_file_list = createShuffledListFiles("chest_xray_data/test/")
# test_image_vector_list = convert_multiple_img_to_vec(test_file_list)
#
# scalar_label_list_test = []
# for file in test_file_list:
#     scalar_label_list_test.append(get_label_from_file(file, 21))
#
# one_hot_label_array_test = []
#
# for scalar1 in scalar_label_list_test:
#     one_hot_label_array_test.append(scalar_to_one_hot(scalar1, num_classes))
#
#
#
#
# input_file_list = createShuffledListFiles("chest_xray_data/train/")
# input_image_vector_list = convert_multiple_img_to_vec(input_file_list)
# scalar_label_list = []
#
# for file in input_file_list:
#     scalar_label_list.append(get_label_from_file(file, 22))
#
# one_hot_label_array = []
#
# for scalar in scalar_label_list:
#     one_hot_label_array.append(scalar_to_one_hot(scalar, num_classes))
#
# # print(one_hot_label_array)
# print(len(input_image_vector_list))
# batch_x, batch_y = get_batches(1, input_image_vector_list, one_hot_label_array, num_classes)
# print(np.array(batch_x), np.array(batch_y))

# print(np.array(batch_x).shape)
# print(np.array(batch_y).shape)
#
# with tf.Session() as sess:
#
#     sess.run(init)
#
#     for step in range(1, epochs+1):
#
#         batch_x, batch_y = get_batches(batch_size, input_image_vector_list, one_hot_label_array)
#
#         #print(np.array(batch_x[0]), np.array(batch_y[0]))
#
#         sess.run(train_op, feed_dict={X: np.array(batch_x), Y: np.array(batch_y), keep_prob: 0.85})
#
#         if step % display_step == 0 or step == 1:
#
#             loss, acc = sess.run([loss_op, accuracy], feed_dict={X: np.array(batch_x), Y: np.array(batch_y), keep_prob: 0.85})
#             print("Step " + str(step) + ", Minibatch Loss= " +
#                   "{:.4f}".format(loss) + ", Training Accuracy= " +
#                   "{:.3f}".format(acc))
#             print("Testing Accuracy: ", sess.run(accuracy, feed_dict={X: np.array(test_image_vector_list),
#                                                                       Y: np.array(one_hot_label_array_test), keep_prob: 1.0}))
#     save_path = saver.save(sess, "/home/student/PycharmProjects/KairoM/network_model.ckpt")
#
#     print("Optimization Finished!")



# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.



with tf.Session() as sess1:
  # Restore variables from disk.
    saver.restore(sess1, "/home/student/PycharmProjects/KairoM/network_model.ckpt")
    print("Model restored.")
    input = np.array(img_to_vector("./online_tester_images/normal_test_1.jpg"), dtype="f")
    input_reshaped = np.reshape(input, (1, 2500))
    output = sess1.run(tf.abs(neural_net(input_reshaped)), feed_dict={keep_prob: 1.0})
    print(output)
    output1 = sess1.run(tf.nn.softmax(output, 1))
    print(output1)
    output2 = sess1.run(tf.argmax(output1, 1))
    print(output2)

    confidence = "%.2f" % (np.amax(output1) * 100)

    if output2[0] == 1:
        print("This patient may have pneumonia. Confidence =  " + confidence + " %")
    else:
        print("This patient may not have pneumonia. Confidence =  " + confidence + " %")
#
#
#
#
