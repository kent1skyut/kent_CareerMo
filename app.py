from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import numpy as np
from wtforms import RadioField
from flask_wtf import FlaskForm
from flask_wtf.csrf import CSRFProtect
from wtforms.validators import InputRequired
import joblib  # Added for model loading
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import mysql.connector
from flask_mysqldb import MySQL
import random
from wtforms import StringField, PasswordField, validators
import bcrypt

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'  # Set your secret key

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'career_assessment'

mysql = mysql.connector.connect(
    host=app.config['MYSQL_HOST'],
    user=app.config['MYSQL_USER'],
    password=app.config['MYSQL_PASSWORD'],
    database=app.config['MYSQL_DB']
)

cursor = mysql.cursor()


csrf = CSRFProtect(app)

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

# Create and train the decision tree model
model = DecisionTreeClassifier()
model.fit(X, y_encoded)

# Save the trained model
joblib.dump(model, 'decision_tree_model.joblib')


# Define the AssessmentForm with a dynamic field for questions
class AssessmentForm(FlaskForm):
    pass  # The form fields will be dynamically added later

# Define a function to load questions from the CSV file
def load_questions_from_csv():
    questions_df = pd.read_csv('question_data.csv')
    questions = questions_df.to_dict('records')
    return questions


def random_color():
    # Generate a random color in hexadecimal format
    color = "#{:02X}{:02X}{:02X}".format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return color

# Define the UserRegistrationForm
class UserRegistrationForm(FlaskForm):
    username = StringField('Username', validators=[validators.InputRequired()])
    email = StringField('Email', validators=[validators.InputRequired(), validators.Email()])
    password = PasswordField('Password', validators=[validators.InputRequired(), validators.Length(min=8)])

# Register route
@app.route('/register', methods=['GET', 'POST'])
@csrf.exempt
def register():
    form = UserRegistrationForm(request.form)
    if request.method == 'POST' and form.validate():
        username = form.username.data
        email = form.email.data
        password = form.password.data.encode('utf-8')

        # Hash the password
        hashed_password = bcrypt.hashpw(password, bcrypt.gensalt())

        # Insert the user data into the 'users' table
        cursor = mysql.connection.cursor()
        cursor.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)", (username, email, hashed_password))
        mysql.connection.commit()
        cursor.close()

        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

# Login route
@app.route('/login', methods=['GET', 'POST'])
@csrf.exempt
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password'].encode('utf-8')

        cursor = mysql.cursor()
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        cursor.close()

        if user and bcrypt.checkpw(password, user['password'].encode('utf-8')):
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password', 'error')
    return render_template('login.html')

@app.route('/logout')
@csrf.exempt
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/search_job', methods=['GET', 'POST'])
@csrf.exempt
def search_job():
    if request.method == 'POST':
        search_query = request.form.get('search_query')

        # Perform a database query to retrieve job information based on the search query
        cursor.execute("SELECT * FROM occupation_data WHERE title LIKE %s OR description LIKE %s", (f'%{search_query}%', f'%{search_query}%'))
        job_results = cursor.fetchall()

        # Create a list of dictionaries with job data and row colors
        row_colors = ['job-color-even', 'job-color-odd']
        job_results_with_colors = [{'data': job, 'color': row_colors[i % 2]} for i, job in enumerate(job_results)]

        # Render the template with the search results
        return render_template('job_search_results.html', job_results=job_results_with_colors)

    return render_template('index.html')



@app.route('/decision_tree')
def decision_tree():
    # Load the saved decision tree model
    loaded_model = joblib.load('decision_tree_model.joblib')

    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(25, 20))

    # Extract unique class names (career clusters)
    unique_career_clusters = y.unique()

    # Create a custom list of class names with proper formatting
    class_names = [f"{i}: {cls}" for i, cls in enumerate(unique_career_clusters)]

    # Define feature names for your questions (question1 to question80)
    feature_names = [f"question{i + 1}" for i in range(80)]

    # Plot the Decision Tree with the custom class names
    plot_tree(loaded_model, feature_names=feature_names, class_names=class_names, filled=True, ax=ax)

    # Save the plot to an image file (optional)
    plt.savefig('decision_tree2.png')

    # Render an HTML template to display the Decision Tree graph
    return render_template('decision_tree.html')


@app.route('/assess', methods=['GET', 'POST'])
def assess():
    form = AssessmentForm(request.form)

    if request.method == 'POST' and form.validate():
        # Accept user input for assessment responses
        user_responses = []

        # Get user responses from the form
        for field_name in form.data:
            if field_name.startswith('question_'):
                response = form[field_name].data
                user_responses.append(response)

        if len(user_responses) == 0:
            # Handle the case where no responses were provided
            return "Please provide responses to the assessment questions."


        # Preprocess user responses
        scaled_user_responses = scaler.transform(np.array(user_responses).reshape(1, -1))

        # Use the trained model to predict all careers' responses
        predicted_encoded_careers = model.predict(X)
        predicted_careers = label_encoder.inverse_transform(predicted_encoded_careers)

        # Calculate the percentage of matching answers for each career
        matching_percentages = []
        for career_responses in X:
            matching_percentage = np.sum(scaled_user_responses == career_responses) / len(user_responses) * 100
            matching_percentages.append(matching_percentage)

        # Find the top recommended careers
        recommended_career_indices = np.argsort(matching_percentages)[::-1]  # Sort in descending order
        top_n_recommendations = 5  # Get the top 5 recommendations

        # Create a list of recommended careers and their matching percentages
        recommended_careers = []
        for i in range(top_n_recommendations):
            recommended_career_index = recommended_career_indices[i]
            recommended_career = predicted_careers[recommended_career_index]
            recommended_percentage = matching_percentages[recommended_career_index]
            recommended_careers.append((recommended_career, recommended_percentage))

        true_careers = label_encoder.inverse_transform(y_encoded)
        # Commenting out accuracy calculation since it's not needed for recommendations
        true_careers = label_encoder.inverse_transform(y_encoded)

        # Calculate precision and recall scores
        precision = precision_score(true_careers, predicted_careers, average='micro')
        recall = recall_score(true_careers, predicted_careers, average='micro')
        accuracy = accuracy_score(true_careers, predicted_careers)
        f1 = f1_score(true_careers, predicted_careers, average='micro')

        # Render an HTML template to display the results
        return render_template('results.html', recommended_careers=recommended_careers, accuracy=accuracy,
                               precision=precision, recall=recall, f1=f1)
    # If it's a GET request, render the assessment form
    questions = load_questions_from_csv()

    # Dynamically add radio fields for each question
    for question in questions:
        field_name = f'question_{question["question_id"]}'
        setattr(AssessmentForm, field_name,
                RadioField(label=question['question_text'], choices=[('1', 'Yes'), ('0', 'No')],
                           validators=[InputRequired()]))

    # Initialize the form
    form = AssessmentForm()

    return render_template('assessment.html', form=form, questions=questions)

if __name__ == '__main__':
    app.run(debug=True)
