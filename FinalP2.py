import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix , classification_report,accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder



#LOADING THE DATA
data_directory = "F:\Project\credit_risk_dataset.csv\credit_risk_dataset.csv"
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
    


'''Now we use one hot encoding for converting'''
encoder= OneHotEncoder(sparse_output=False)
one_hot_encoded_data=encoder.fit_transform(data_frame[categorical])
one_hot_encoded_data_frame=pd.DataFrame(
                           one_hot_encoded_data,
                           columns=encoder.get_feature_names_out(categorical))

data_frame['person_emp_length']=data_frame['person_emp_length'].fillna(data_frame['person_emp_length'].median())
data_frame['loan_int_rate']=data_frame['loan_int_rate'].fillna(data_frame['loan_int_rate'].mean())
'''Then we scale the numeric ones'''
scaler= StandardScaler()
scaled_data=scaler.fit_transform(data_frame[numerical])
scaled_data_frame=pd.DataFrame(
                  scaled_data,
                  columns=numerical
                )

#Preparing data for the model
X= pd.concat([one_hot_encoded_data_frame,scaled_data_frame],axis=1)
Y=data_frame['loan_status']
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.2,random_state=42)

#The model
model_obj=LogisticRegression(penalty='l1',solver='liblinear',C=1.0,random_state=42)
model_obj.fit(X_train,Y_train)

#The results
coefficients=pd.Series(model_obj.coef_[0],
                       index=X_train.columns
                    )
coefficients_dataframe=pd.DataFrame({
                    'Coefficient':coefficients,
                    'Absolute value': coefficients.abs(),
                    'Relationship':coefficients.apply(lambda x: 'Increases default'if x > 0 else ('Decreases default'))
                })
#print(coefficients_dataframe)
threshold=0.3
y_pred_new=(model_obj.predict_proba (X_test)[:,1]>=threshold).astype(int)

#Perfomance metrics
#y_pred = model_obj.predict(X_test)
accuracy=accuracy_score(Y_test,y_pred_new)
print("Accuracy of the model  :" + str (accuracy*100))

'''Get confusion matrix to see where the model messed up '''
confusion_matrix_c= confusion_matrix(Y_test,y_pred_new, labels=[0,1])
confusion_matrix_decorated=pd.DataFrame(confusion_matrix_c,
                                        index=['Actual non default','Actual default'], 
                                        columns=['Predicted non default', 'Predicted default'])
print(confusion_matrix_decorated)

