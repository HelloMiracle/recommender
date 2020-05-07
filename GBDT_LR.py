from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
oneHot=OneHotEncoder()
oneHot.fit([[1,1,2],[2,2,2],[3,1,3],[4,5,4]])
print(oneHot.transform([[1,1,2],[2,2,2],[3,1,3],[4,5,4]]).toarray())




random_seed=10
class GBDT_LR():
    def __init__(self,data,label,gbdt_name):
        self.gbdt_set=['xgboost','gbdt','lgb']
        self.gbdt_name=gbdt_name
        self.data=data
        self.label=label
        self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(self.data,self.label,train_size=0.7,random_state=random_seed)
        self.gbdt=self.init_gbdt()

    def init_gbdt(self):
        if self.gbdt_name == 'xgboost':
            gbdt = XGBClassifier()
        elif self.gbdt_name=='gbdt':
            gbdt=GradientBoostingClassifier()
        elif self.gbdt_name=='lgb':
            gbdt=LGBMClassifier()
        else:
            print('no valid gbdt model')
        return gbdt


    def gbdt_train(self):
        self.gbdt.fit(self.x_train,self.y_train)
    def gbdt_predict(self):
        self.gbdt_predict =self.gbdt.predict_proba(self.x_test)
    def cal_auc(self):
        gbdt_auc=roc_auc_score(self.y_test,self.gbdt_predict)

    def LR(self):
        gbdt_encoder=OneHotEncoder()
        self.lr=LogisticRegression()
        self.x_train_leafs=self.gbdt.apply(self.x_train)
        self.x_test_leafs=self.gbdt.apply(self.x_test)
        gbdt_encoder.fit(self.x_train_leafs)
        x_train_encoder=gbdt_encoder.transform(self.x_train_leafs)
        x_test_encoder = gbdt_encoder.transform(self.x_test_leafs)
        self.lr.fit(x_train_encoder,self.y_train)
        self.gbdt_lr_predict=self.lr.predict_proba(x_test_encoder)
        gbdt_lr_auc=roc_auc_score(self.y_test,self.gbdt_lr_predict)
        print('基于gbdt编码后的LR AUC值:{:.2f}'.format(gbdt_lr_auc))
        lr2=LogisticRegression()
        lr2.fit(self.x_train)
        lr_predict=lr2.predict_proba(self.x_test)
        lr_auc=roc_auc_score(self.x_test,lr_predict)
        print('LR AUC值:{:.2f}'.format(lr_auc))








    def init_Lr(self):
        pass

