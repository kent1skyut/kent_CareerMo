from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import joblib

# Load the training data
training_data = pd.read_csv('training_data.csv')

# Define the feature columns (questions 1-10)
feature_columns = ['question1', 'question2 ', 'question3', 'question4', 'question5',
                   'question6', 'question7', 'question8', 'question9', 'question10']

# Define the target column (career)
target_column = 'career'

# Split the data into features and target
X = training_data[feature_columns]
y = training_data[target_column]

# Initialize and train the decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X, y)

# Save the trained model to a file
joblib.dump(clf, 'decision_tree_models.pkl')
