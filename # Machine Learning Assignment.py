# Section 1: Importing Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Section 2: Loading the Dataset
file_path = r'C:\Users\Zirian\Downloads\insurance.csv'  # Use raw string to avoid unicode escape issues
insurance_data = pd.read_csv(file_path)

# Section 3: Data Exploration
summary_statistics = insurance_data.describe(include='all')
missing_values = insurance_data.isnull().sum()

# Section 4: Printing Data Exploration Results
print("Summary Statistics:\n", summary_statistics)
print("\nMissing Values:\n", missing_values)

# Section 5: Correlation Analysis
# Convert categorical columns to numerical
insurance_data_encoded = insurance_data.copy()
insurance_data_encoded['sex'] = insurance_data_encoded['sex'].map({'male': 0, 'female': 1})
insurance_data_encoded['smoker'] = insurance_data_encoded['smoker'].map({'yes': 1, 'no': 0})
insurance_data_encoded = pd.get_dummies(insurance_data_encoded, columns=['region'], drop_first=True)

correlation_matrix = insurance_data_encoded.corr()
medical_cost_correlation = correlation_matrix["medicalCost"].sort_values(ascending=False)
print("\nCorrelation with Medical Cost:\n", medical_cost_correlation)

# Section 6: Preparing Data for Simple Linear Regression
X_age = insurance_data_encoded[['age']]
X_bmi = insurance_data_encoded[['bmi']]
X_children = insurance_data_encoded[['children']]
y = insurance_data_encoded['medicalCost']

# Section 7: Splitting the Data into Training and Testing Sets
X_age_train, X_age_test, y_train, y_test = train_test_split(X_age, y, test_size=0.2, random_state=42)
X_bmi_train, X_bmi_test, y_train, y_test = train_test_split(X_bmi, y, test_size=0.2, random_state=42)
X_children_train, X_children_test, y_train, y_test = train_test_split(X_children, y, test_size=0.2, random_state=42)

# Section 8: Creating and Training the Models
model_age = LinearRegression()
model_bmi = LinearRegression()
model_children = LinearRegression()

model_age.fit(X_age_train, y_train)
model_bmi.fit(X_bmi_train, y_train)
model_children.fit(X_children_train, y_train)

# Section 9: Making Predictions
y_age_pred = model_age.predict(X_age_test)
y_bmi_pred = model_bmi.predict(X_bmi_test)
y_children_pred = model_children.predict(X_children_test)

# Section 10: Evaluating the Models
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, mae, r2

age_mse, age_mae, age_r2 = evaluate_model(y_test, y_age_pred)
bmi_mse, bmi_mae, bmi_r2 = evaluate_model(y_test, y_bmi_pred)
children_mse, children_mae, children_r2 = evaluate_model(y_test, y_children_pred)

print("\nSimple Linear Regression Results:")
print(f"Age: MSE = {age_mse}, MAE = {age_mae}, R-squared = {age_r2}")
print(f"BMI: MSE = {bmi_mse}, MAE = {bmi_mae}, R-squared = {bmi_r2}")
print(f"Children: MSE = {children_mse}, MAE = {children_mae}, R-squared = {children_r2}")

# Section 11: Preparing Data for Multivariate Regression
X_top3 = insurance_data_encoded[['age', 'bmi', 'children']]
X_all = insurance_data_encoded[['age', 'bmi', 'children', 'sex', 'smoker', 'region_northwest', 'region_southeast', 'region_southwest']]

# Section 12: Splitting the Data for Multivariate Regression Models
X_top3_train, X_top3_test, y_train, y_test = train_test_split(X_top3, y, test_size=0.2, random_state=42)
X_all_train, X_all_test, y_train, y_test = train_test_split(X_all, y, test_size=0.2, random_state=42)

# Section 13: Creating and Training Multivariate Regression Models
model_top3 = LinearRegression()
model_all = LinearRegression()

model_top3.fit(X_top3_train, y_train)
model_all.fit(X_all_train, y_train)

# Section 14: Making Predictions with Multivariate Models
y_top3_pred = model_top3.predict(X_top3_test)
y_all_pred = model_all.predict(X_all_test)

# Section 15: Evaluating the Multivariate Regression Models
top3_mse, top3_mae, top3_r2 = evaluate_model(y_test, y_top3_pred)
all_mse, all_mae, all_r2 = evaluate_model(y_test, y_all_pred)

print("\nMultivariate Regression Results:")
print(f"Top 3 Predictors: MSE = {top3_mse}, MAE = {top3_mae}, R-squared = {top3_r2}")
print(f"All Predictors: MSE = {all_mse}, MAE = {all_mae}, R-squared = {all_r2}")

