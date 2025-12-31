import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#STEP 1 : Load data
data_directory = "F:\\Project\\StudentsPerformance.csv"
data=pd.read_csv(data_directory)

#STEP 2 : CONVERT CATEGORICAL DATA TO NUMERICAL DATA
data['test preparation course']=(data['test preparation course'].replace({'completed':1,'none':0}).astype(int))

#STEP 3 : GROUP DATA ACCORDINGLY
'''feel free to change the titles of the scores'''
grouped_data= data.groupby(['test preparation course'])[['math score']]

#STEP 4:CHECK SIGNIFICANCE OF THE RELATIONSHIP USING A TWO SAMPLE T TEST
prepared=data[data['test preparation course'] == 1]['math score']
unprepared=data[data['test preparation course'] == 0]['math score']
t_stat, p_val=stats.ttest_ind(prepared,unprepared,equal_var=False)
print("P value :" + str(p_val))

# STEP 5 : BUILD THE MODEL
X= data[['test preparation course']]
Y= data['math score']

'''Split the dataset into training and testing'''
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
'''instintiate the model object'''
model = LinearRegression()
'''fit the training data into the model oject'''
model.fit(X_train,y_train)
'''y predictioons after the model is done i.e student marks based on actual input '''
y_pred = model.predict(X_test)
''''print the r^2 value'''
print("R^2 score on test set:", model.score(X_test, y_test)*100)

