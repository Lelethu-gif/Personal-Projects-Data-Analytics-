import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

data_directory = "F:\\Project\\credit_risk_dataset.csv\\credit_risk_dataset.csv"
data=pd.read_csv(data_directory)
data['cb_person_default_on_file'] = data['cb_person_default_on_file'].replace({'Y':1, 'N':0}).astype(int)
numeric_features = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
binary_features = ['cb_person_default_on_file']  # already encoded

categorical_features = ['person_home_ownership', 'loan_intent', 'loan_grade']

# -------------------------------
# 4️⃣ One-hot encode categorical variables
# -------------------------------
encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_cats = encoder.fit_transform(data[categorical_features])

encoded_df = pd.DataFrame(
    encoded_cats,
    columns=encoder.get_feature_names_out(categorical_features),
    index=data.index
)

# -------------------------------
# 5️⃣ Final feature matrix
# -------------------------------

X = pd.concat([data[numeric_features + binary_features], encoded_df], axis=1)
Y = data['loan_status']
# Drop rows with NaN values
X_clean = X.dropna()
y_clean = Y[X_clean.index]  # make sure target matches cleaned features

# -------------------------------
# 6️⃣ Train/test split
# -------------------------------

X_train, X_test, Y_train, Y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

# Fit Lasso Logistic Regression
model = LogisticRegression(
    penalty='l1',
    solver='liblinear',
    C=0.1,
    random_state=42,
    max_iter=1000
)
model.fit(X_train, Y_train)
# -------------------------------
# 8️⃣ Extract important features
# -------------------------------
coef_df = pd.DataFrame({
    'feature': X.columns,
    'coefficient': model.coef_[0]
})

important_features = coef_df[coef_df['coefficient'] != 0]
important_features['abs_coef'] = important_features['coefficient'].abs()
important_features = important_features.sort_values(by='abs_coef', ascending=False)

#print("Important features and their coefficients:\n")
#print(important_features)

# Train/test split on cleaned data

# Keep only features with positive coefficients
positive_features = important_features[important_features['coefficient'] > 1]

# Sort by coefficient magnitude
positive_features = positive_features.sort_values(by='coefficient', ascending=False)

print(positive_features[['feature', 'coefficient']])
