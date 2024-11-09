from flask import Flask, render_template, request
import numpy as np
import ml  # Import your ml.py module

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the user via the form
    year_of_release = int(request.form['year_of_release'])
    developer = request.form['developer']  
    genre = request.form['genre']  
    youtube_likes = int(request.form['youtube_likes'])
    twitter_followers = int(request.form['twitter_followers'])

    # Preprocess the inputs (e.g., encode categorical variables)
    input_data = np.array([year_of_release, youtube_likes, twitter_followers]).reshape(1, -1)

    # Use the prediction function from ml.py
    prediction = ml.predict(input_data)
    
    return render_template('index.html', prediction_text=f'Prediction: {prediction[0]}')

if __name__ == '__main__':
    app.run(debug=True)
