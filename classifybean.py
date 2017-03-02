import tensorflow as tf
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
import random
import sys

reload = True

def get_padding_size(image):
    h, w = image.shape
    longest_edge = max(h, w)
    top, bottom, left, right = (0, 0, 0, 0)
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass
    return top, bottom, left, right


def read_data(img_path,image_data,label_data):
    for filename in os.listdir(img_path):
        if filename.endswith('.bmp'):
            filepath = os.path.join(img_path, filename)
            image = cv2.imread(filepath)
            #cv2.imshow("d",image)
            '''top, bottom, left, right = get_padding_size(image)
            image_pad = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            '''
            image = cv2.resize(image, (48, 48))
            image_data.append(image)
            label_data.append(img_path)
os.chdir("E:\\咖啡\\生豆素材\\classifycolor")
print(os.getcwd())

def read():

    well_image_path = 'well'
    bad_image_path = 'bad'
    image_data = []
    label_data = []
    read_data(bad_image_path,image_data,label_data)
    read_data(well_image_path,image_data,label_data)
    image_data = np.array(image_data)
    label_data = np.array([[0,1] if label == 'well' else [1,0] for label in label_data])

    train_x, test_x, train_y, test_y = train_test_split(image_data, label_data, test_size=0.05, random_state=random.randint(0, 100))

    # image (height=64 width=64 channel=3)
    train_x = train_x.reshape(train_x.shape[0], 48, 48, 3)
    test_x = test_x.reshape(test_x.shape[0], 48, 48, 3)

    # nomalize
    train_x = (train_x.astype('float32')-128) / 128.0
    test_x = (test_x.astype('float32') - 128) / 128.0

    print(len(train_x), len(train_y))
    print(len(test_x), len(test_y))
    return train_x,train_y,test_x,test_y
#############################################################
batch_size = 100


X = tf.placeholder(tf.float32, [None, 48, 48, 3])  # 图片大小64x64 channel=3
Y = tf.placeholder(tf.float32, [None, 2])

keep_prob_5 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)


def save(filename, sess, num,saver):
    path = os.getcwd()
    os.chdir(".//"+filename)
    #saver = tf.train.Saver()
    saver.save(sess, "./bean.model", global_step=num)
    os.chdir(path)


def bean_cnn():
    W_c1 = tf.Variable(tf.random_normal([5, 5, 3, 32], stddev=0.0001))
    b_c1 = tf.Variable(tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(X, W_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob_5)


    W_c2 = tf.Variable(tf.random_normal([5, 5, 32, 32], stddev=0.01))
    b_c2 = tf.Variable(tf.random_normal([32]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, W_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.avg_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob_5)


    W_c3 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
    b_c3 = tf.Variable(tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, W_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.avg_pool(conv3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob_5)

    # Fully connected layer
    W_d = tf.Variable(tf.random_normal([6*6*64, 100], stddev=0.1))
    b_d = tf.Variable(tf.random_normal([100]))
    dense = tf.reshape(conv3, [-1, W_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, W_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob_75)

    '''
    W_d2 = tf.Variable(tf.random_normal([128, 128], stddev=0.01))
    b_d2 = tf.Variable(tf.random_normal([128]))
    dense2 = tf.reshape(dense, [-1, W_d2.get_shape().as_list()[0]])
    dense2 = tf.nn.relu(tf.add(tf.matmul(dense2, W_d2), b_d2))
    dense2 = tf.nn.dropout(dense2, keep_prob_75)
    '''


    W_out = tf.Variable(tf.random_normal([100, 2], stddev=0.1))
    b_out = tf.Variable(tf.random_normal([2]))
    out = tf.add(tf.matmul(dense, W_out), b_out)
    W1=W_d
    #W2=W_d2
    W3=W_out
    return out, W1, W3
def train_cnn():
    j = 1
    lastacc = 0
    output, W1, W3= bean_cnn()

    tf.summary.histogram("layer1/weights", W1)
    #tf.summary.histogram("layer2/weights", W2)
    tf.summary.histogram("layer3/weights", W3)
    loss_l2 = 0.0001*tf.nn.l2_loss(W1)+0.1*tf.nn.l2_loss(W3)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, Y)+loss_l2)
    loss_L2 = tf.reduce_mean(loss_l2)
    # 自动减少learning_rate
    # cur_step = tf.Variable(0, trainable=False)  # count the number of steps taken.
    # starter_learning_rate = 8e-5
    # learning_rate = tf.train.exponential_decay(starter_learning_rate, cur_step, 100000, 0.96, staircase=True)
    # optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=cur_step)
    optimizer = tf.train.AdamOptimizer(learning_rate=5e-5).minimize(loss)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1)), tf.float32))
    tf.summary.scalar("l2", loss_l2)
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)
    merged_summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        if reload:    # 如果reload为true 重新加载旧的模型参数
            saver.restore(sess, tf.train.latest_checkpoint('.\old11\model36'))
        else:         # 否则 初始化所有参数
            sess.run(tf.global_variables_initializer())
        train_x, train_y, test_x, test_y = read()
        num_batch = len(train_x) // batch_size
        summary_writer = tf.summary.FileWriter('e:/logs', graph=tf.get_default_graph())

        for e in range(10000):
            for i in range(num_batch):
                batch_x = train_x[i*batch_size: (i+1)*batch_size]
                batch_y = train_y[i*batch_size: (i+1)*batch_size]
                _, loss_, summary, accuracy_, loss_l2_ = sess.run([optimizer, loss, merged_summary_op, accuracy, loss_L2], feed_dict={X: batch_x, Y: batch_y, keep_prob_5: 0.75, keep_prob_75: 0.5})

                summary_writer.add_summary(summary, e*num_batch+i)
                print(e*num_batch+i, "loss:", str(loss_))
                print(e*num_batch+i, "accuracy:", accuracy_)

                if (e*num_batch+i) % 100 == 0:
                    acc = accuracy.eval({X: test_x, Y: test_y, keep_prob_5:1.0, keep_prob_75: 1.0})
                    print(e*num_batch+i,"准确度:", str(acc))
                    print(e*num_batch+i,"L2:", str(loss_l2_))
                    # save model

                    if acc > lastacc and acc > 0.85:
                        save("model"+str(j), sess, e*num_batch+i,saver)
                        j += 1
                        lastacc = acc
                    if acc > 0.99:
                        save("model"+str(j), sess, e*num_batch+i,saver)
                        j += 1
                        sys.exit(0)
        save("model"+str(j), sess, e*num_batch+i,saver)
        j += 1
        sys.exit(0)
train_cnn()
