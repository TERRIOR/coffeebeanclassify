import cv2
import os
import numpy as np

import tensorflow as tf


def large(rect,size):
    rect[2]=rect[2]+size
    rect[3]=rect[3]+size
    return rect
def bean_cnn():
    W_c1 = tf.Variable(tf.random_normal([5, 5, 3, 16], stddev=0.01))
    b_c1 = tf.Variable(tf.random_normal([16]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(X, W_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob_5)


    W_c2 = tf.Variable(tf.random_normal([5, 5, 16, 32], stddev=0.01))
    b_c2 = tf.Variable(tf.random_normal([32]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, W_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob_5)


    '''W_c3 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
    b_c3 = tf.Variable(tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, W_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob_5)
    '''
    # Fully connected layer
    W_d = tf.Variable(tf.random_normal([8*8*32, 128], stddev=0.01))
    b_d = tf.Variable(tf.random_normal([128]))
    dense = tf.reshape(conv2, [-1, W_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, W_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob_75)


    W_d2 = tf.Variable(tf.random_normal([128, 128], stddev=0.01))
    b_d2 = tf.Variable(tf.random_normal([128]))
    dense2 = tf.reshape(dense, [-1, W_d2.get_shape().as_list()[0]])
    dense2 = tf.nn.relu(tf.add(tf.matmul(dense2, W_d2), b_d2))
    dense2 = tf.nn.dropout(dense2, keep_prob_75)

    W_out = tf.Variable(tf.random_normal([128, 2], stddev=0.01))
    b_out = tf.Variable(tf.random_normal([2]))
    out = tf.add(tf.matmul(dense2, W_out), b_out)
    return out
def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
def draw_rect(img, rect, color):
    cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), color, 2)


def is_well_bean(image):

    img=image.reshape(32, 32, 3).astype('float32')
    res = sess.run(predict, feed_dict={X: [img/255.0], keep_prob_5:1.0, keep_prob_75: 1.0})
    if res[0] == 1:
        return True
    else:
        return False


def color_vote( rect, iswell, scale, count):

    x=rect[0]//scale
    y=rect[1]//scale
    #print(x,y)
    global resultmat, votemat,drawmat
    if count >= 40:
        print("vote",votemat[x,y])
        if totalmat[x, y] > 10:
            drawmat[x,y] = 1
        else:
            drawmat[x,y] = 0
        if votemat[x, y] >= 30:
            resultmat[x, y] = 1

        elif votemat[x,y] < 30:
            resultmat[x, y] = 0

    else:
        totalmat[x,y] = totalmat[x, y]+1
        if iswell is True:
            votemat[x, y] = votemat[x, y]+1
    vote=resultmat[x, y]
    return count, vote, drawmat[x,y]



if __name__ == "__main__":
    import sys, getopt
    scale=30
    count=0
    global resultmat, votemat, totalmat, drawmat
    resultmat=np.ones([640//scale,480//scale],'int32')
    votemat=np.zeros([640//scale,480//scale],'int32')
    totalmat=np.zeros([640//scale,480//scale],'int32')
    drawmat=np.zeros([640//scale,480//scale],'int32')
    X = tf.placeholder(tf.float32, [None, 32, 32, 3])  # 图片大小64x64 channel=3
    Y = tf.placeholder(tf.float32, [None, 2])
    keep_prob_5 = tf.placeholder(tf.float32)
    keep_prob_75 = tf.placeholder(tf.float32)
    output = bean_cnn()
    predict = tf.argmax(output, 1)
    saver = tf.train.Saver()
    sess = tf.Session()
    os.chdir('E:\咖啡\生豆素材\classifycolor')
    print(os.listdir())
    saver.restore(sess, tf.train.latest_checkpoint('.\old10\model4'))
    args, video_src= getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    args = dict(args)
    cascade_fn = args.get('--cascade',"E:/opencv/opencv/build/x64/vc12/bin/data/coffeenewdata/cascade.xml")
    cascade = cv2.CascadeClassifier(cascade_fn)
    cam = cv2.VideoCapture(1)
    while True:
        _, img = cam.read()
        if img is None:
            sys.exit(0)
        #print(img.shape)
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #cv2.imshow("gray",gray)
        rects = detect(gray, cascade)
        vis = img.copy()
        count+=1
        if count>=41:
            count=0
            votemat[:,:] = 0
            totalmat[:, :] = 0
        for rect in rects:
            #large(rect,1)
            save_mat = img[rect[1]:rect[3], rect[0]:rect[2]]

            if save_mat is not None:
                #save_mat =cv2.GaussianBlur(save_mat,(3,3),0)
                save_mat = cv2.resize(save_mat,(32,32))
                #save_mat = cv2.equalizeHist(save_mat)
                #cv2.waitKey(1000)
                #cv2.imshow("hist",save_mat)
                is_well = is_well_bean(save_mat)
                #print(is_well)
                count, is_well_ave, if_draw = color_vote(rect, is_well, scale, count)
                if if_draw == 1:
                    if is_well_ave == 1:
                        draw_rect(vis, rect, (0, 0, 255))
                    elif is_well_ave == 0:
                        draw_rect(vis, rect, (0, 0, 0))

        cv2.imshow('img', vis)
        key = cv2.waitKey(10) & 0xff
        if key == 27:
            sys.exit(0)
    cam.release()


