

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import tree
import matplotlib.pyplot as plt

# Load the dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data'
column_names = ['Status', 'Duration', 'Credit_history', 'Purpose', 'Credit_amount', 'Savings', 'Employment', 'Installment_rate', 'Personal_status', 'Other_debtors', 'Residence_since', 'Property', 'Age', 'Other_installment_plans', 'Housing', 'Existing_credits', 'Job', 'Liable', 'Telephone', 'Foreign_worker', 'Default']
data = pd.read_csv(url, delimiter=' ', header=None, names=column_names)

# Preprocessing (encoding categorical variables, handling missing values, etc.)
# This is a simplified example, further preprocessing may be required based on the dataset specifics
data = pd.get_dummies(data, drop_first=True)

# Define features and target variable
X = data.drop('Default', axis=1)
y = data['Default']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Decision Tree model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

# Plot the Decision Tree
plt.figure(figsize=(20,10))
tree.plot_tree(clf, filled=True, feature_names=X.columns.tolist(), class_names=['No Default', 'Default'])
plt.show()



