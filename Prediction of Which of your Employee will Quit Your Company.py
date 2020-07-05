#Importing packages
import numpy as np
import pandas as pd

#Importing dataset
#Loading HR data
hr_df=pd.read_csv('./datasets/hr_data.csv')

#HR dataframe attributes
hr_df
hr_df.shape
hr_df.size
hr_df.info()
hr_df['department'].unique()
hr_df['salary'].unique()

#Loading our Employee Satisfaction Data
s_df=pd.read_excel('./datasets/employee_satisfaction_evaluation.xlsx')

#Employee Satisfaction Dataframe
s_df

#Joining both dataframes
main_df= hr_df.set_index('employee_id').join(s_df.set_index('EMPLOYEE #'))

#Main datafram attributes
main_df=main_df.reset_index()
main_df
main_df.info()
main_df[main_df.isnull().any(axis=1)]
main_df.describe()

#Data preprocessing
main_df.fillna(main_df.mean(),inplace=True)
main_df[main_df.isnull().any(axis=1)]
main_df.loc[main_df['employee_id']==1340]
main_df.drop(columns='employee_id',inplace=True)
main_df
main_df.groupby('department').sum()
main_df.groupby('department').mean()
main_df['left'].value_counts()

#Data Visualization
#Importing packages
import matplotlib.pyplot as plt
import seaborn as sns

#Plotter function
def plot_corr(df,size=10):    
    corr=df.corr()
    fig,ax=plt.subplots(figsize=(size,size))
    ax.legend()
    cax=ax.matshow(corr)
    fig.colorbar(cax)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical')
    plt.yticks(range(len(corr.columns)), corr.columns)
    

plot_corr(main_df)

plt.bar(x=main_df['left'],height=main_df['satisfaction_level'])
sns.barplot(x='left',y='satisfaction_level',data=main_df)
sns.barplot(x='promotion_last_5years',y='satisfaction_level',data=main_df,hue='left')
sns.pairplot(main_df,hue='left')

#Data Processing for decision tree
#Importing packages 
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

k=le.fit_transform(main_df['salary'])
main_df['salary_num']=k
main_df.loc[main_df['salary']=='high']
main_df.drop(['salary'],axis=1,inplace=True)

z=le.fit_transform(main_df['department'])
main_df['department_num']=z
main_df.loc[main_df['department']=='IT']
main_df.drop(['department'],axis=1,inplace=True)

X=main_df.drop(['left'],axis=1)
y=main_df['left']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,)

# Model Classification
# Decision Tree
#Importing packages 
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)

prediction_dt=dt.predict(X_test)

#Prediction results
prediction_dt
y_test

#Accuracy
accuracy_dt=accuracy_score(y_test,prediction_dt)*100
accuracy_dt

#Predicting with custom data
Catagory=['Employee will stay','Employee will Leave']
custom_dt=[[1,500,3,6,0,0.90,0.89,1,8]]
print(int(dt.predict(custom_dt)))
Catagory[int(dt.predict(custom_dt))]

dt.feature_importances_
feature_importance=pd.DataFrame(dt.feature_importances_,index=X_train.columns,columns=['Importance']).sort_values('Importance',ascending=False)
feature_importance
X_train

# Data Processing of KNN
#Importing packages 
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

sc=StandardScaler().fit(X_train)
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)

knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_std,y_train)

#Prediction results
prediction_knn=knn.predict(X_test_std)
prediction_knn
y_test

#Accuracy
accuracy_knn=accuracy_score(y_test,prediction_knn)*100
accuracy_knn

k_range=range(1,26)
scores={}
scores_list=[]

#Prediction Visualization graph
for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_std,y_train)
    prediction_knn=knn.predict(X_test_std)
    scores[k]=accuracy_score(y_test,prediction_knn)*100
    scores_list.append(accuracy_score(y_test,prediction_knn))

scores
scores_list

plt.plot(k_range,scores_list)
X_test.head(1)

X_knn=np.array([[20,500,10,6,0,0.10,0.30,1,8]])
X_knn_std=sc.transform(X_knn)
X_knn_std
X_knn_prediction=knn.predict(X_knn_std)
X_knn_prediction

Catagory[int(dt.predict(custom_dt))]

algorithms=['Decision Tree','KNN']
scores=[accuracy_dt,accuracy_knn]
plt.xlabel("Algorithms")
plt.ylabel("Accuracy Score")
sns.barplot(algorithms,scores)
plt.show()