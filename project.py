import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('placement.csv')
#print(df.head())

# 1. Pre-Process + EDA + Features Selection
# 2. Extract Input/Output Coulumns
# 3. Train Test Split
# 4. Train the Model
# 5. Evaluate the Model
# 6. Deploy the Model  
df=df.iloc[:,1:]
#print(df.info())#check if there null values
#print(df.head())


#plt.scatter(df['cgpa'], df['iq'],c=df['placement'])
#plt.show()  
x=df.iloc[:,0:2]
y=df.iloc[:,-1]
#print(y)
#print(y.shape)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1)
#print(x_train)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
#print(x_train)
x_test=scaler.transform(x_test)
#print(x_test)

#Training the Model
from sklearn.linear_model import LogisticRegression
cls=LogisticRegression()
cls.fit(x_train,y_train)
 
 #Checking Model Prediction
#print(cls.predict(x_test))
y_pred =(cls.predict(x_test))

#check model Predictioin with Given Dataset Prediction
#print(y_test)
import joblib

# After training

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))


joblib.dump(cls, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
