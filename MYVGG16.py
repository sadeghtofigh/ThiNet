import tensorflow as tf
import numpy as np
import pickle
import ChangeFilter

# Unzip the dataset
def unpickle(file):
   '''Load byte data from file'''
   with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='latin-1')
   return data

# Loading dataset information
def load_cifar10_data(data_dir):
 '''Return train_data, train_labels, test_data, test_labels
 The shape of data is 32 x 32 x3'''
 train_data = None
 X_labels = []

 for i in range(1, 6):
  data_dic = unpickle(data_dir + "/data_batch_{}".format(i))
  if i == 1:
   train_data = data_dic['data']
  else:
   train_data = np.vstack((train_data, data_dic['data']))
  X_labels += data_dic['labels']

 test_data_dic = unpickle(data_dir + "/test_batch")
 test_data = test_data_dic['data']
 Y_labels = test_data_dic['labels']

 train_data = train_data.reshape((len(train_data), 3, 32, 32))
 train_data = np.rollaxis(train_data, 1, 4)
 X_labels = np.array(X_labels)

 test_data = test_data.reshape((len(test_data), 3, 32, 32))
 test_data = np.rollaxis(test_data, 1, 4)
 Y_labels = np.array(Y_labels)

 return train_data, X_labels, test_data, Y_labels
data_dir = 'C:/Users/s_tofigh/Desktop/Python/Sample Python codes-other people/CIF' # Directory of downloaded dataset on PC
train_data, X_labels, test_data, Y_labels = load_cifar10_data(data_dir)


test_labels = np.zeros((len(Y_labels), 10))
for i in range(len(Y_labels)):
    a= np.zeros((1,10))
    k=Y_labels[i]
    a[:,k]= 1
    test_labels[i] = a


tf.reset_default_graph()
# import the graph from the file
imported_graph = tf.train.import_meta_graph('model.ckpt.meta', clear_devices=True)

#saver = tf.train.Saver(save_relative_paths=True)
#saver.restore(sess, tf.train.latest_checkpoint('C:/Users/s_tofigh/Desktop/Python/My_CNN_CIFAR10/CNN/Data Analysis/Semi_Prun'))


# Reading Dataset
with tf.Session() as sess:
    imported_graph.restore(sess, tf.train.latest_checkpoint("./"))
    wc1 = sess.run('W0:0')
with tf.Session() as sess:
    imported_graph.restore(sess, tf.train.latest_checkpoint("./"))
    wc2 = sess.run('W1:0')
with tf.Session() as sess:
    imported_graph.restore(sess, tf.train.latest_checkpoint("./"))
    wc3 = sess.run('W2:0')
with tf.Session() as sess:
    imported_graph.restore(sess, tf.train.latest_checkpoint("./"))
    wc4 = sess.run('W3:0')
with tf.Session() as sess:
    imported_graph.restore(sess, tf.train.latest_checkpoint("./"))
    wc5 = sess.run('W4:0')
with tf.Session() as sess:
    imported_graph.restore(sess, tf.train.latest_checkpoint("./"))
    wc6 = sess.run('W5:0')
with tf.Session() as sess:
    imported_graph.restore(sess, tf.train.latest_checkpoint("./"))
    wc7 = sess.run('W6:0')
with tf.Session() as sess:
    imported_graph.restore(sess, tf.train.latest_checkpoint("./"))
    wd1 = sess.run('W13:0')
with tf.Session() as sess:
    imported_graph.restore(sess, tf.train.latest_checkpoint("./"))
    wd2 = sess.run('W14:0')
with tf.Session() as sess:
    imported_graph.restore(sess, tf.train.latest_checkpoint("./"))
    wd3 = sess.run('W15:0')
with tf.Session() as sess:
    imported_graph.restore(sess, tf.train.latest_checkpoint("./"))
    L_out = sess.run('W16:0')
with tf.Session() as sess:
    imported_graph.restore(sess, tf.train.latest_checkpoint("./"))
    bc1 = sess.run('B0:0')
with tf.Session() as sess:
    imported_graph.restore(sess, tf.train.latest_checkpoint("./"))
    bc2 = sess.run('B1:0')
with tf.Session() as sess:
    imported_graph.restore(sess, tf.train.latest_checkpoint("./"))
    bc3 = sess.run('B2:0')
with tf.Session() as sess:
    imported_graph.restore(sess, tf.train.latest_checkpoint("./"))
    bc4 = sess.run('B3:0')
with tf.Session() as sess:
    imported_graph.restore(sess, tf.train.latest_checkpoint("./"))
    bc5 = sess.run('B4:0')
with tf.Session() as sess:
    imported_graph.restore(sess, tf.train.latest_checkpoint("./"))
    bc6 = sess.run('B5:0')
with tf.Session() as sess:
    imported_graph.restore(sess, tf.train.latest_checkpoint("./"))
    bc7 = sess.run('B6:0')
with tf.Session() as sess:
    imported_graph.restore(sess, tf.train.latest_checkpoint("./"))
    bd1 = sess.run('B13:0')
with tf.Session() as sess:
    imported_graph.restore(sess, tf.train.latest_checkpoint("./"))
    bd2 = sess.run('B14:0')
with tf.Session() as sess:
    imported_graph.restore(sess, tf.train.latest_checkpoint("./"))
    bd3 = sess.run('B15:0')
with tf.Session() as sess:
    imported_graph.restore(sess, tf.train.latest_checkpoint("./"))
    b_out = sess.run('B16:0')
def conv_net_VGG16(x, A,B,C):
    # First Block
    conv1 = ChangeFilter.conv2d(x, wc1, bc1)
    #conv1=tf.nn.dropout(conv1, rate=0.5)
    conv2 = ChangeFilter.conv2d(conv1, wc2, bc2)
    #conv2 = tf.nn.dropout(conv2, rate=0.5)
    conv2 = ChangeFilter.maxpool2d(conv2, k=2)
    # Second Block
    conv3 = ChangeFilter.conv2d(conv2, wc3, bc3)
    #conv3 = tf.nn.dropout(conv3, rate=0.5)
    conv4 = ChangeFilter.conv2d(conv3, wc4, bc4)
    #conv4 = tf.nn.dropout(conv4, rate=0.5)
    conv4 = ChangeFilter.maxpool2d(conv4, k=2)
    # Third Block
    conv5 = ChangeFilter.conv2d(conv4, wc5, bc5)
    #conv5 = tf.nn.dropout(conv5, rate=0.5)
    conv6 = ChangeFilter.conv2d(conv5, A, C)
    #conv6 = tf.nn.dropout(conv6, rate=0.5)
    conv7 = ChangeFilter.conv2d(conv6, B, bc7)
    #conv7 = tf.nn.dropout(conv7, rate=0.5)
    conv7 = ChangeFilter.maxpool2d(conv7, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv7, [-1, wd1.shape[0]])
    fc1 = tf.add(tf.matmul(fc1, wd1), bd1)
    fc1 = tf.nn.relu(fc1)
    #fc1 = tf.nn.dropout(fc1, rate=0.5)
    fc2 = tf.add(tf.matmul(fc1, wd2), bd2)
    fc2 = tf.nn.relu(fc2)
    #fc2 = tf.nn.dropout(fc2, rate=0.5)
    fc3 = tf.add(tf.matmul(fc2, wd3), bd3)
    fc3 = tf.nn.relu(fc3)
    #fc3 = tf.nn.dropout(fc3, rate=0.5)
    # Output, class prediction
    # finally we multiply the fully connected layer with the weights and add a bias term.
    out = tf.add(tf.matmul(fc3, L_out), b_out)
    return out


def TestNet(x,y,test_batch_size, A,B,C):

    test_acc_2 = 0
    pred = conv_net_VGG16(x, A,B,C)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:

        for tbatch in range(10):
            test_batch_x = test_data[tbatch * test_batch_size:min((tbatch + 1) * \
                                                                  test_batch_size, len(test_data))]
            test_batch_y = test_labels[tbatch * test_batch_size:min((tbatch + 1) * \
                                                                    test_batch_size, len(test_labels))]
            TA, valid_loss = sess.run([accuracy, cost], feed_dict={x: test_batch_x, y: test_batch_y})
            test_acc_2 = test_acc_2 + TA
                #print("Test Iteration: " + str(tbatch + 1) + "/10  ======================>> Test Accuracy: " + str(TA))
        test_acc_2 = test_acc_2 / 10


    return test_acc_2



X1 = [[2,2,2], [5,8,9]]
X2 = [[2,2,2], [2,4,9]]
X3 = [[2,2,2], [3,4,11]]
X4 = [[1,1,2], [1,1,9]]
X5 = [[1,1,2], [1,1,9]]
X6 = [[1,1,2], [1,1,9]]

def Net_Filter_Selection(input_size):
    Nor1 = np.zeros(((input_size*(len(X1))),64))
    Nor2 = np.zeros(((input_size*(len(X2))),64))
    Nor3 = np.zeros(((input_size*(len(X3))),128))
    Nor4 = np.zeros(((input_size*(len(X4))),128))
    Nor5 = np.zeros(((input_size*(len(X5))),256))
    Nor6 = np.zeros(((input_size*(len(X6))),256))
    fir = np.zeros((input_size, 64))
    for sbatch in range (input_size):
            print('Counter =',sbatch)
            x = (train_data[sbatch].reshape(1,32,32,3)).astype('float32')
            conv1 = ChangeFilter.conv2d(x, wc1, bc1)
            Nor1[sbatch*(len(X1)):(sbatch+1)*(len(X1)),:] = ChangeFilter.filter_selec(X1 , conv1,wc2)
            conv2 = ChangeFilter.conv2d(conv1, wc2,bc2)
            conv2 = ChangeFilter.maxpool2d(conv2, k=2)
            Nor2[sbatch*(len(X2)):(sbatch+1)*(len(X2)),:] = ChangeFilter.filter_selec(X2 , conv2,wc3)
            # Second Block
            conv3 = ChangeFilter.conv2d(conv2, wc3,bc3)
            Nor3[sbatch*(len(X3)):(sbatch+1)*(len(X3)),:] = ChangeFilter.filter_selec(X3 , conv3,wc4)
            conv4 = ChangeFilter.conv2d(conv3, wc4,bc4 )
            conv4 = ChangeFilter.maxpool2d(conv4, k=2)
            Nor4[sbatch*(len(X4)):(sbatch+1)*(len(X4)),:] = ChangeFilter.filter_selec(X4 , conv4,wc5)
            # Third Block
            conv5 = ChangeFilter.conv2d(conv4, wc5, bc5)
            Nor5[sbatch*(len(X5)):(sbatch+1)*(len(X5)),:] = ChangeFilter.filter_selec(X5 , conv5,wc6)
            conv6 = ChangeFilter.conv2d(conv5, wc6, bc6)
            Nor6[sbatch*(len(X6)):(sbatch+1)*(len(X6)),:] = ChangeFilter.filter_selec(X6 , conv6,wc7)
    posi_layer_ind_1 = ChangeFilter.arrange(Nor1)
    posi_layer_ind_2 = ChangeFilter.arrange(Nor2)
    posi_layer_ind_3 = ChangeFilter.arrange(Nor3)
    posi_layer_ind_4 = ChangeFilter.arrange(Nor4)
    posi_layer_ind_5 = ChangeFilter.arrange(Nor5)
    posi_layer_ind_6 = ChangeFilter.arrange(Nor6)
    return posi_layer_ind_1,posi_layer_ind_2,posi_layer_ind_3,posi_layer_ind_4,posi_layer_ind_5,posi_layer_ind_6
