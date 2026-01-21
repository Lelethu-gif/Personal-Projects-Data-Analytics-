import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix , classification_report,accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder



#LOADING THE DATA
data_directory = "F:\Project credit\credit_risk_dataset.csv"
data=pd.read_csv(data_directory)
data_frame= pd.DataFrame(data)

#CONVERTING CATEGORICAL TO NUMERICAL VARIABLES
numerical=[ 'person_age','person_income','person_emp_length','loan_amnt','loan_int_rate','loan_percent_income','cb_person_cred_hist_length']
categorical= ['person_home_ownership','loan_intent','loan_grade','cb_person_default_on_file']

#DESCRIPRIVE ANALYSIS
'''for var in numerical:
    average= data_frame.groupby('loan_status')[var].mean()
    plt.figure(figsize=(5,5))
    plt.bar(['Non-default','Default'],average)
    plt.ylabel(f'Average {var}')
    plt.xlabel('Loan status')
    plt.title(f'Average {var} by loan status')
    plt.show()
'''
##Training th& testing the model#
X = data_frame[categorical + numerical]
Y= data_frame['loan_status']
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

##Handling missing values##
X_train['person_emp_length']=X_train['person_emp_length'].fillna(X_train['person_emp_length'].median())
X_train['loan_int_rate']=X_train['loan_int_rate'].fillna(X_train['loan_int_rate'].mean())

X_test['person_emp_length']=X_test['person_emp_length'].fillna(X_train['person_emp_length'].median())
X_test['loan_int_rate']=X_test['loan_int_rate'].fillna(X_train['loan_int_rate'].mean())

##ONE HOT ENCODING##
encoder= OneHotEncoder(sparse_output=False)
one_hot_encoded_training_data=encoder.fit_transform(X_train[categorical])
one_hot_encoded_testing_data=encoder.transform(X_test[categorical])
one_hot_encoded_training_data_frame=pd.DataFrame(
                           one_hot_encoded_training_data,
                           columns=encoder.get_feature_names_out(categorical))
one_hot_encoded_testing_data_frame=pd.DataFrame(
                           one_hot_encoded_testing_data,
                           columns=encoder.get_feature_names_out(categorical))


##Then we scale the numeric ones##
scaler= StandardScaler()
X_train_scaled_data=scaler.fit_transform(X_train[numerical])
X_test_scaled_data=scaler.transform(X_test[numerical])
scaled_train_data_frame=pd.DataFrame(
                  X_train_scaled_data,
                  columns=numerical
                )
scaled_test_data_frame=pd.DataFrame(
                  X_test_scaled_data,
                  columns=numerical
                )

#Final preparation data for the model

X_train_final= pd.concat([one_hot_encoded_training_data_frame,scaled_train_data_frame],axis=1)
X_test_final=pd.concat([one_hot_encoded_testing_data_frame,scaled_test_data_frame],axis=1)
                           
model_obj=LogisticRegression(penalty='l1',solver='liblinear',C=1.0,random_state=42)
model_obj.fit(X_train_final,y_train)


y_pred_new=(model_obj.predict_proba (X_test_final)[:,1]>=threshold).astype(int)

#Perfomance metrics
accuracy=accuracy_score(y_test,y_pred_new)
print("Accuracy of the model  :" + str (accuracy*100))

'''Get confusion matrix to see where the model messed up '''
confusion_matrix_c= confusion_matrix(y_test,y_pred_new, labels=[0,1])
confusion_matrix_decorated=pd.DataFrame(confusion_matrix_c,
                                        index=['Actual non default','Actual default'], 
                                        columns=['Predicted non default', 'Predicted default'])
print("confusion matrix : ")
print(confusion_matrix_decorated)

