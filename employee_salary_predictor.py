import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Load data from CSV
# Map income: <=50K -> 40000, >50K -> 60000

def income_to_salary(income):
    return 40000 if income.strip() == '<=50K' else 60000

# Read only relevant columns for simplicity
usecols = ['age', 'education', 'educational-num', 'occupation', 'hours-per-week', 'gender', 'income']
df = pd.read_csv('adult 3.csv', usecols=usecols)

# Drop rows with missing values
for col in ['education', 'occupation', 'gender']:
    df = df[df[col] != '?']

df['salary'] = df['income'].apply(income_to_salary)

# Extract options BEFORE get_dummies
education_options = sorted(df['education'].unique())
occupation_options = sorted(df['occupation'].unique())
gender_options = sorted(df['gender'].unique())

# Encode categorical variables
categorical_cols = ['education', 'occupation', 'gender']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# 2. Split data
X = df.drop(['income', 'salary'], axis=1)
y = df['salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Evaluate
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# 5. Prompt user for input using numbers
age = int(input('Enter age: '))
educational_num = int(input('Enter years of education (educational-num): '))
hours_per_week = int(input('Enter hours worked per week: '))

print("Education options:")
for i, opt in enumerate(education_options):
    print(f"  {i+1}. {opt}")
education_idx = int(input('Choose education (enter number): ')) - 1
education = education_options[education_idx]

print("Occupation options:")
for i, opt in enumerate(occupation_options):
    print(f"  {i+1}. {opt}")
occupation_idx = int(input('Choose occupation (enter number): ')) - 1
occupation = occupation_options[occupation_idx]

print("Gender options:")
for i, opt in enumerate(gender_options):
    print(f"  {i+1}. {opt}")
gender_idx = int(input('Choose gender (enter number): ')) - 1
gender = gender_options[gender_idx]

# 6. Build example input for prediction
example = {
    'age': age,
    'educational-num': educational_num,
    'hours-per-week': hours_per_week,
}
for col in X.columns:
    if col not in example:
        example[col] = 0
# Set the relevant one-hot columns to 1 (if present)
if f'education_{education}' in X.columns:
    example[f'education_{education}'] = 1
if f'occupation_{occupation}' in X.columns:
    example[f'occupation_{occupation}'] = 1
# For gender, handle drop_first logic
if 'gender_Male' in X.columns or 'gender_Female' in X.columns:
    for g in ['Male', 'Female']:
        colname = f'gender_{g}'
        if colname in X.columns:
            example[colname] = 1 if gender == g else 0
example_df = pd.DataFrame([example])[X.columns]
predicted_salary = model.predict(example_df)[0]
print(f"Predicted salary for the entered employee profile: ${predicted_salary:.2f}") 