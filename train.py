import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score

# Load the training data
data = pd.read_csv('training_data1.csv')

# Separate features (questions) and target (career)
X = data.iloc[:, :-1]  # Features
y = data['career']     # Target

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Encode the target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Create and train the decision tree model with regularization
model = DecisionTreeClassifier(max_depth=5, min_samples_split=5, min_samples_leaf=2)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Decode the predicted career labels to get recommended careers
recommended_careers = label_encoder.inverse_transform(y_pred)

# Calculate precision and recall on the test set
precision = precision_score(y_test, y_pred, average='micro')
recall = recall_score(y_test, y_pred, average='micro')

print(f"Precision on test data: {precision}")
print(f"Recall on test data: {recall}")

# Accept user input for responses to questions
user_responses = []  # Initialize an empty list to store user responses

# Assuming there are 80 questions in the dataset
for i in range(1, 81):
    response = float(input(f"Enter your response to question {i}: "))
    user_responses.append(response)

# Standardize and encode user input
user_input = scaler.transform([user_responses])
user_encoded = model.predict(user_input)[0]

# Decode the predicted career label to get the recommended career
recommended_career = label_encoder.inverse_transform([user_encoded])[0]

print(f"Based on your responses, the recommended career is: {recommended_career}")
