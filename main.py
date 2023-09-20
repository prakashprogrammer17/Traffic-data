import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from save_load import *

#data_gen
data=pd.read_csv('C:\\Users\\ASUS\\Desktop\\region_traffic.csv')
data=data.drop(columns=['region_name','road_category_name', 'road_category_description'])
data['label']=data['road_category_id'].apply(lambda x:1 if 5>x else 0)
label=data['label']
label=np.array(label)
data=data.drop(columns=['label','road_category_id'])


#feat_extration
data['mean']=data.mean(axis=1)
data['median']=data.median(axis=1)
data['skew']=data.skew(axis=1)
data['kurtosis']=data.kurtosis(axis=1)
data['std']=data.std(axis=1)

#feature selecation
pca=PCA(n_components=8)
feat=pca.fit_transform(data)


X_train,X_test,Y_train,Y_test=train_test_split(feat,label,test_size=0.3)
save('X_train',X_train)
save('X_test',X_test)
save('Y_train',Y_train)
save('Y_test',Y_test)
