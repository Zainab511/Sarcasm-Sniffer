from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load the LR model and vectorizer
vectorizer = joblib.load('vectorizer.joblib')
lr_model = joblib.load('lr_model.joblib')


def predict_sarcasm_lr(text):
    # Vectorize the input text
    text_vect = vectorizer.transform([text])

    # Make prediction using the loaded model
    prediction = lr_model.predict(text_vect)

    # Return 'Sarcastic' for 1 and 'Non-sarcastic' for 0
    return 'Sarcastic' if prediction[0] == 1 else 'Non-sarcastic'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/predict_lr', methods=['POST'])
def predict_lr():
    if request.method == 'POST':
        text_to_predict = request.form['text']

        # Get the prediction result
        prediction_result = predict_sarcasm_lr(text_to_predict)

        return jsonify({'prediction': prediction_result})


if __name__ == '__main__':
    app.run(debug=True)
