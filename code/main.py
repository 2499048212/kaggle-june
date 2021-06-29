from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
train_path = 'C:\\Users\\24990\\Desktop\\tabular-playground-series-jun-2021\\train.csv'
test_path = 'C:\\Users\\24990\\Desktop\\tabular-playground-series-jun-2021\\test.csv'
subm_path = 'C:\\Users\\24990\\Desktop\\tabular-playground-series-jun-2021\\sample_submission.csv'
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
samp_sub = pd.read_csv(subm_path)
samp_sub = pd.read_csv(subm_path)
target_mass = train_df['target'].value_counts()
values = target_mass.values.tolist()
indexes = target_mass.index.tolist()
fet_set = train_df.drop(labels=['id','target'],axis=1)
def plot_diag_heatmap(data):
    corr = data.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    f, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, mask=mask, cmap='YlGnBu', center=0,square=True, linewidths=1, cbar_kws={"shrink": 1.0})
corr = train_df.iloc[:,1:-1].corr()
for col in corr.columns:
    if ((sum(corr[col])-1)/(len(corr)-1)) <0.06:
        train_df.drop(col,1,inplace=True)
        test_df.drop(col,1,inplace=True)
fig,axes = plt.subplots(1,5,figsize=(24,3))
i=1
for col in train_df.columns[1:-1]:
    sns.boxplot(train_df['target'],train_df[col])
    i+=1
    if i%5==1 and col!=train_df.columns[-2]:
        i=1
        fig,axes = plt.subplots(1,5,figsize=(24,3))
from scipy.stats import iqr
temp_df = train_df

for col in temp_df.columns[1:-1]:
    iqr_val = iqr(temp_df[col])
    q1 = np.quantile(temp_df[col] , 0.25)
    q3 = np.quantile(temp_df[col] , 0.75)
    temp_df = temp_df[temp_df[col]>=q1-1.5*iqr_val]
    temp_df = temp_df[temp_df[col]<=q3+1.5*iqr_val]
from  scipy.stats import zscore
temp_df = train_df

for col in temp_df.columns[1:-1]:
    temp_df['zs'] = np.abs(zscore(temp_df[col]))
    temp_df = temp_df[temp_df['zs'] <= 3.0]
    temp_df.drop('zs' , 1 , inplace = True)
train_df.drop('zs' , 1 , inplace = True)
cleaned_train_df = temp_df
cleaned_train_df.drop('id',1,inplace=True)
idx = test_df['id']
test_df.drop('id',1,inplace=True)
cleaned_train_df.drop_duplicates(inplace=True)
cleaned_train_df = cleaned_train_df.T.drop_duplicates().T
arr = []
for i in range(1,10):
    t_df =temp_df[temp_df['target']=='Class_'+str(i)]
from sklearn.model_selection import train_test_split
def split_data(test_size,data):
    data = data.sample(frac=1)
    x_train = data.drop('target',1)
    y_1 = data['target']
    x_train = x_train
    y_1 = y_1.to_numpy()
    X_train , X_val , y_1 , y_2 = train_test_split( x_train , y_1 ,
                                                         test_size = test_size ,
                                                        random_state =1 ,
                                                        stratify = y_1)
    y_train = []
    y_val = []
    for value in y_1:
        y_train.append(int(value[-1])-1)
    for value in y_2:
        y_val.append(int(value[-1])-1)
    return X_train , X_val , np.array(y_train) , np.array(y_val)
X_train , X_val , y_train , y_val = split_data(0.2,cleaned_train_df)
X_test = test_df[X_train.columns]
from sklearn.preprocessing import StandardScaler as scaler
def scale(train,test,validation):
  sc = scaler()
  columns = train.columns
  train = sc.fit_transform(train)
  test = sc.transform(test)
  validation = sc.transform(validation)

  train = pd.DataFrame(train , columns = columns)
  test = pd.DataFrame(test , columns = columns)
  validation = pd.DataFrame(validation , columns = columns)

  return train , test , validation
X_train , X_test , X_val = scale(X_train , X_test , X_val)
tm = pd.DataFrame(y_train,columns=['x'])
target_mass = tm['x'].value_counts()
values = target_mass.values.tolist()
indexes = target_mass.index.tolist()
from catboost import CatBoostClassifier as cbt
def train_and_predict(model , x_1  , x_2 , x_3 , y_1 , y_2):
    labels = []
    for i in range(9):
        labels.append('Class_'+str(i+1))
    model.fit(x_1 , y_1)
    print('Training Completed..........')
    print('Train Accuracy : ',model.score(x_1,y_1))
    print('Validation Accuracy : ',model.score(x_2 , y_2))
    print('Model Prediction started....')
    y_pred = model.predict_proba(x_3)
    final_df = pd.DataFrame(y_pred , columns = labels)
    final_df = pd.concat([idx,final_df]  , axis = 1)    #uncomment this to find the actual submission files.
    
    return final_df
model = cbt(verbose=0)
submission = train_and_predict(model , X_train , X_val , X_test , y_train , y_val)
submission.to_csv('cbt'+'.csv',index=False)
print('Done')
