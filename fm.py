import tensorflow as tf
import os
from util import *
import numpy as np
class Fm():
    def __init__(self,num_classes,k,batch_size,feature_size,iteration,train_feature,train_label,save_dir):
        self.num_classes=num_classes
        self.k=k
        self.batch_size=batch_size
        self.feature_size=feature_size
        self.iteration=iteration
        self.train_feature=train_feature
        self.train_label=train_label
        self.save_dir=save_dir

    def bulid_model(self):
        self.x=tf.placeholder('float32',[None,self.feature_size])
        self.y=tf.placeholder('float32',[None,self.num_classes])
        with tf.variable_scope('linear_layer'):
            w0=tf.get_variable('w0',shape=[self.num_classes],initializer=tf.zeros_initializer)
            w=tf.get_variable('w',shape=[self.feature_size,self.num_classes],initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.01))
            linear_out=tf.add(tf.matmul(self.x,w),w0)
        with tf.variable_scope('interaction_layer'):
            embeding=tf.get_variable('v',shape=[self.feature_size,self.k],initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.01))
            interaction_out=tf.multiply(0.5
                                        ,tf.reduce_sum(
                                           tf.subtract(
                                               tf.pow(tf.matmul(self.x,embeding),2),
                                               tf.matmul(tf.pow(self.x,2),tf.pow(embeding,2))),axis=1,keepdims=True))
        with tf.variable_scope('out_layer'):
            output=tf.add(linear_out,interaction_out)

        with tf.variable_scope('accuracy'):
            correct_prediction = tf.equal(tf.cast(tf.argmax(output,1), tf.float32), tf.cast(tf.argmax(self.y,1), tf.float32))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            # add summary to accuracy
            tf.summary.scalar('accuracy', self.accuracy)
        if self.num_classes==2:
            y_prob=tf.nn.sigmoid(output)
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=y_prob)
        elif self.num_classes>2:
            y_prob=tf.nn.softmax(output)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=y_prob)
        mean_loss = tf.reduce_mean(cross_entropy)
        self.loss = mean_loss
        tf.summary.scalar('loss', self.loss)
        self.optimizer=tf.train.AdamOptimizer()
        self.train_op=self.optimizer.minimize(self.loss)

    # def next_batch(self):
    #     print('mext+_natch',self.train_feature)
    #     print(self.batch_size)
    #     input_queue = tf.train.slice_input_producer([self.train_feature,self.train_label], shuffle=False)
    #     feature_batch, label_batch = tf.train.batch(input_queue, batch_size=self.batch_size, num_threads=2, capacity=128,allow_smaller_final_batch=True)
    #     return feature_batch,label_batch

    def shuffle_list(self,data):
        num = data[0].shape[0]
        p = np.random.permutation(num)
        return [d[p] for d in data]

    def batch_generator(self,data, batch_size, shuffle=False):
        if shuffle:
            data = self.shuffle_list(data)

        batch_count = 0
        while True:
            if batch_count * batch_size + batch_size > len(data[0]):
                batch_count = 0

                if shuffle:
                    data = self.shuffle_list(data)

            start = batch_count * batch_size
            end = start + batch_size
            batch_count += 1
            yield [d[start:end] for d in data]

    def train(self):
        self.Saver = tf.train.Saver(max_to_keep=100)
        merge = tf.summary.merge_all()
        nums_batch=len(self.train_label)//self.batch_size+1
        init = tf.initialize_all_variables()
        ckpt=tf.train.get_checkpoint_state(self.save_dir)




        with tf.Session() as self.sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(self.sess, coord)
            self.sess.run(init)
            train_writer=tf.summary.FileWriter('./log/train_logs',self.sess.graph)
            if ckpt and ckpt.model_model_checkpoint_path:
                self.Saver.restore(self.sess,ckpt.model_model_checkpoint_path)
                print("加载模型成功"+ckpt.model_model_checkpoint_path)
                global_step=int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            else:
                global_step =0
            for epoch in range(self.iteration):
                for _ in range(nums_batch):
                    #feature_batch, label_batch = self.next_batch()
                    feature, label = next(self.batch_generator([self.train_feature,self.train_label],batch_size=self.batch_size))
                    #feature,label = self.sess.run([feature_batch, label_batch])
                    feed_dic={self.x:feature, self.y:label}
                    loss, accuracy, summary, _ = self.sess.run([self.loss, self.accuracy,merge, self.train_op], feed_dict=feed_dic)
                    train_writer.add_summary(summary, global_step=global_step)
                    global_step+=1
                if epoch % 100 == 0:
                    print('save_model_{}'.format(epoch))
                    self.Saver.save(self.sess, os.path.join(self.save_dir, 'fm'), global_step=global_step)
                print("Epoch {1},  loss = {0:.3g}, accuracy={2:.3g}".format(loss, epoch + 1,accuracy))
            coord.request_stop()
            coord.join(threads)
if __name__=='__main__':
    x_train,  x_test,y_train, y_test = load_data()
    print(x_train)
    # initialize the model
    num_classes = 5
    lr = 0.01
    batch_size = 128
    k = 40
    #reg_l1 = 2e-2
    #reg_l2 = 0
    feature_length = x_train.shape[1]
    save_dir=r'./model/'
    # initialize FM model
#def __init__(self,num_classes,k,batch_size,feature_size,iteration,train_feature,train_label):
    model = Fm(num_classes, k, batch_size, feature_length,1000,x_train,y_train,save_dir=save_dir)
    model.bulid_model()
    model.train()
    # build graph for model


    #saver = tf.train.Saver(max_to_keep=5)




