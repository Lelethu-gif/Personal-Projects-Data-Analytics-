import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

#///
# THIS MODEL WILL PREDICT WHETHER A STUDENT WILL FAIL OR PASS GIVEN THE FACT THAT THEY HAVE PREPARED OR NOT
# USING THEIR AVERAGE ACROSS THE THREE SUBJECTS 
# \\\
 
#STEP 1 : Load data
data_directory = "F:\\Project\\StudentsPerformance.csv"
data=pd.read_csv(data_directory)

#STEP 2 : CONVERT CATEGORICAL DATA TO NUMERICAL DATA
data['test preparation course']=(data['test preparation course'].replace({'completed':1,'none':0}).astype(int))
data['average']= ((data['math score']+ data['reading score']+ data['writing score'])/3)
data['final_mark']=np.where(data['average']>= 50 ,'Pass','Fail')


#STEP 3: BUILDING THE MODEL
X= data[['test preparation course']]
Y= data['final_mark']

'''Split the dataset into training and testing'''
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=42)

'''Instintiate model logistic regression model'''
model= linear_model.LogisticRegression()

'''Fit the training data into the object model'''
model.fit(X_train,y_train)

'''Get the predictions'''
y_pred = model.predict(X_test)

'''get probabilities for the positive class'''
y_pred_probabilities= model.predict_proba(X_test)

#fail = 1-y_pred_probabilities
#print ( "Probability of passing " + "  |   "+ "    Probability of failing")
#print ( "--------------------- " + "   -|---------  "+ "   ----------------------")
#for p in fail:
   
 #   print( "            "  +str(round(p[0],2))+"         " + "|"+str(round(p[1],2)))

'''Get accuracy score'''
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy of the model  :" + str (accuracy*100))

'''Get confusion matrix to see where the model messed up '''
confusion_matrix_c= confusion_matrix(y_test,y_pred, labels=['Pass','Fail'])
confusion_matrix_decorated=pd.DataFrame(confusion_matrix_c,index=['Actual Pass','Actual Fail'], columns=['Predicted Pass', 'Predicted Fail'])
print(confusion_matrix_decorated)