# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from patsy import dmatrices
import joblib

# Load the dataset
data = pd.read_csv('census_data.csv')

# Drop rows with missing values
data = data.dropna(how='any')

# Define the formula for the model
formula = 'greater_than_50k ~ age + workclass + education + marital_status + occupation + race + gender + hours_per_week + native_country'

# Prepare the data for the model
y, X = dmatrices(formula, data=data, return_type='dataframe')

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest model
clf = RandomForestClassifier(n_estimators=100, oob_score=True, min_samples_split=5, min_samples_leaf=2)

# Train the model on the training set
clf.fit(X_train, y_train.greater_than_50k)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
conf_matrix = confusion_matrix(y_test.greater_than_50k, y_pred)
print("Confusion Matrix:\n", conf_matrix)

print("\nClassification Report:\n", classification_report(y_test.greater_than_50k, y_pred))

# Feature importance
model_ranks = pd.Series(clf.feature_importances_, index=X_train.columns, name='Importance').sort_values(ascending=False)
print(model_ranks)

# Visualize feature importance
plt.figure(figsize=(20, 7))
model_ranks.plot(kind='barh')
plt.title("Variable Ranking")
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

# Save the trained model
joblib.dump(clf, 'random_forest_income_predictor.pkl')

# Save the classification report
with open('classification_report.txt', 'w') as f:
    f.write(classification_report(y_test.greater_than_50k, y_pred))
