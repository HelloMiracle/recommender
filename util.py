import pandas as pd
import numpy as np
def load_data():
    user_header=['user_id','age','gender','occupation','zipcode']
    df_user=pd.read_csv('./data/u.user',sep='|',names=user_header)

    item_header = ['item_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation', 'Children',
            'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
            'Thriller', 'War', 'Western']
    df_item = pd.read_csv('data/u.item', sep='|', names=item_header, encoding = "ISO-8859-1")
    df_item=df_item.drop(['title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown'],axis=1)
    df_user['age'] = pd.cut(df_user['age'], [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                            ,labels=['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90','90-100']
                            )
    df_user=pd.get_dummies(df_user,columns=['gender','age','occupation'])
    df_user=df_user.drop(['zipcode'],axis=1)

    user_features = df_user.columns.values.tolist()
    movie_features = df_item.columns.values.tolist()
    # print(user_features,movie_features)
    features=user_features+movie_features
    rate_header = ['user_id', 'item_id', 'rating', 'timestamp']
    df_train = pd.read_csv('data/ua.base', sep='\t', names=rate_header)
    df_train=df_train.drop(['timestamp'],axis=1)
    df_train = df_train.merge(df_user, on='user_id', how='left')
    df_train = df_train.merge(df_item, on='item_id', how='left')

    df_test = pd.read_csv('data/ua.test', sep='\t', names=rate_header)
    df_test = df_test.merge(df_user, on='user_id', how='left')
    df_test = df_test.merge(df_item, on='item_id', how='left')
    train_labels = pd.get_dummies(df_train['rating'])
    train_labels.columns=['rate_'+str(name) for name in train_labels.columns]
    test_labels = pd.get_dummies(df_test['rating'])
    test_labels.columns = ['rate_' + str(name) for name in test_labels.columns]
    #print(df_train[features].values)
    return df_train[features].values,df_test[features].values,train_labels,test_labels
load_data()