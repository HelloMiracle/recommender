import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf
# load data to build real rating matrix
file_name = './data/user_item_rating.txt'
rating_matrix = np.loadtxt(file_name, dtype=bytes).astype(float)
user_num = rating_matrix.shape[0]
item_num = rating_matrix.shape[1]

# initialize user and item matrix with random float between -1 and 1(not included)
feature_num = 3
# user_matrix = np.random.random_sample((user_num, feature_num))
# item_matrix = random.random_sample((item_num, feature_num))

class MF():
    def __init__(self,rate,feature_size,lam,learning_rate):
        self.k=feature_size
        self.rate=rate
        self.user_num=rate.shape[0]
        self.item_num=rate.shape[1]
        self.lam=lam
        self.learning_rate=learning_rate
    def build_embeding(self):
        self.user=np.random.random_sample((self.user_num,self.k))
        self.item=np.random.random_sample((self.item_num, self.k))

    def train(self,epoch):
        for iter in range(epoch):
            for i in range(self.user_num):
                for j in range(self.item_num):
                    if self.rate[i,j]!=0:
                        e_ui = self.rate[i,j] - np.dot(self.user[i, :] , self.item[j, :])
                        self.user[i, :] += self.learning_rate * (e_ui * self.item[j, :] - self.lam * self.user[i, :])
                        self.item[j, :] += self.learning_rate * (e_ui * self.user[i, :] - self.lam * self.item[j, :])
            iter_mse=self.cal_mse()
            print(iter,iter_mse)
        return self.user,self.item

    def cal_mse(self):
        self.rating_predict=np.dot(self.user,self.item.transpose())
        fliter=self.rate>0
        predict_filterd=np.multiply(self.rating_predict,fliter)
        diff=np.multiply((predict_filterd-self.rate),(predict_filterd-self.rate))
        self.mse=np.sum(diff)/np.count_nonzero(self.rate)
        return self.mse



mf=MF(rating_matrix,feature_num,0.1,0.01)
mf.build_embeding()
user,item=mf.train(100)
print(mf.cal_mse())
rating_predict=np.dot(user,item.transpose())
#print(rating_matrix)
np.savetxt('./data/rating_predict.txt',rating_predict,fmt='%.2f')
#if __name__=='__main__':



        #label=tf.placeholder('float32',[self.user_num,self.item_num])
        #UserEmbed=tf.get_variable('userembed',shape=[self.user_num,self.k])
        #ItemEmbed = tf.get_variable('itemembed', shape=[self.item_num, self.k])


