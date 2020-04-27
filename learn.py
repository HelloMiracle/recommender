import numpy as np
import tensorflow as tf
from util import *
import numpy as np
class ML():
    def __init__(self,batch_size,iteration,train_feature,train_label):
        self.batch_size=batch_size
        self.iteration=iteration
        self.train_feature=train_feature
        self.train_label=train_label
    def next_batch(self):
        print(self.train_feature)
        input_queue = tf.train.slice_input_producer([self.train_feature, self.train_label], shuffle=False)
        feature_batch, label_batch = tf.train.batch(input_queue, batch_size=self.batch_size, num_threads=1,capacity=256)
        return feature_batch, label_batch
    def train(self):
        with tf.Session() as sess:
            t, label_batch = self.next_batch()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)
            for j in range(8):
                x,out = sess.run([t,label_batch])
                print(x,out)
            coord.request_stop()
            coord.join(threads)
if __name__=='__main__':
    batch_size = 128
    x_train, y_train, x_test, y_test = load_data()
    model = ML(batch_size,1000,x_train,y_train)
    model.train()



def loadExData3():
    return [[2,0,4,4,4,0,0,0,0,0],
            [0, 0, 4, 4, 4, 0, 0, 0, 0, 0],
            [0, 0, 1, 3, 0, 0, 0, 0, 0, 5],
            [3, 0, 4, 4, 0, 0, 1, 0, 4, 0],
            [5, 0, 0, 0, 4, 0, 0, 0, 0, 0]]
def standEst(dataMat,user,sinmeas,item):
    n=np.shape(dataMat)[1]
    simTotal=0.0
    ratSimTotal=0.0
    for j in range(n):
        userRating=dataMat[user,j]
        if userRating==0:
            continue
        overlap=np.nonzero(np.logical_and(dataMat[:,item].A>0,dataMat[:,j].A>0))
        if len(overlap)==0:
            similarity=0
        else:
            similarity=sinmeas(dataMat[overlap,item],dataMat[overlap,j])
        simTotal+=similarity
        ratSimTotal+=similarity*userRating
    if simTotal==0:
        return 0
    else:
        return ratSimTotal/simTotal
