from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Initialize Flask app
app = Flask(__name__)

# Load dataset and train model
df = pd.read_csv("vedha_final_data.csv")

# Preprocess the data
label_encoder = LabelEncoder()
df['Grades'] = label_encoder.fit_transform(df['Grades'])

# Define features (X) and target (y)
X = df.drop(columns=['Student ID', 'Grades', 'Sum', 'Total_marks'])
y = df['Grades']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Initialize and train the model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Test model accuracy
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Training Accuracy: {accuracy:.4f}")


@app.route('/')
def index():
    """Render the main page with the input form."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle form submission and make predictions."""
    try:
        # Get user inputs from the form
        roll_no = request.form['roll_no']
        ct1 = float(request.form['ct1'])
        ct2 = float(request.form['ct2'])
        sport = float(request.form['sport'])
        cca = float(request.form['cca'])
        attend = float(request.form['attend'])
        em = float(request.form['em'])

        # Calculate CT average
        ct_avg = (ct1 + ct2) / 2

        # Create a DataFrame for the input
        user_input = pd.DataFrame({
            'CT_1_Marks_20': [ct1],
            'CT_2_Marks_20': [ct2],
            'Class_test_20_marks': [ct_avg],
            'sports_10 marks': [sport],
            'Extra_curricular': [cca],
            'Attendence': [attend],
            'Exam_marks_50 marks': [em]
        })

        # Predict grade
        user_pred = clf.predict(user_input)
        predicted_grade = label_encoder.inverse_transform(user_pred)
        print(roll_no)
        print(predicted_grade[0])

        # Return prediction result
        return render_template(
            'result.html',
            roll_no=roll_no,
            grade=predicted_grade[0]
)
    except Exception as e:
        return f"An error occurred: {e}"


if __name__ == '__main__':
    app.run(debug=True)
