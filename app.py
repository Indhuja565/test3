from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Dictionary mapping model names to their respective pickle file paths
model_files = {
    "Logistic Regression": "Logistic_Regression.pkl",
    "Decision Tree": "Decision_Tree.pkl",
    "Random Forest": "Random_Forest.pkl",
    "SVM": "SVM.pkl",
    "KNN": "KNN.pkl"
}

# Load models from their respective pickle files
models = {}
for name, path in model_files.items():
    with open(path, 'rb') as file:
        models[name] = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    logs = []
    predictions = {}
    chart_data = {}

    try:
        input_data = [float(request.form[key]) for key in request.form]
        logs.append("Input successfully received and converted.")
    except ValueError:
        logs.append("Invalid input! Please enter numbers.")
        return render_template('index.html', logs=logs)

    input_array = np.array(input_data).reshape(1, -1)

    for name, model in models.items():
        pred = model.predict(input_array)[0]
        label = "At Risk" if pred == 1 else "No Risk"
        predictions[name] = label
        chart_data[name] = pred
        logs.append(f"{name} predicted: {label}")

    return render_template('index.html',
                           predictions=predictions,
                           logs=logs,
                           chart_data=chart_data)

if __name__ == '__main__':
    app.run(debug=True)
