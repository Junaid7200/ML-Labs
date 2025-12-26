from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.datasets import load_iris, load_digits, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

app = Flask(__name__)
os.makedirs('static', exist_ok=True)
# UPLOAD_FOLDER = 'uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def load_dataset(name):
    if name == 'iris':
        data = load_iris()
        return data.data, data.target
    elif name == 'digits':
        data = load_digits()
        return data.data, data.target
    elif name == 'wine':
        data = load_wine()
        return data.data, data.target

def load_model(name):
    if name == 'logistic':
        return LogisticRegression(max_iter=200)
    elif name == 'knn':
        return KNeighborsClassifier(n_neighbors=3)
    elif name == 'svm':
        return SVC()


@app.route('/', methods=['GET', 'POST'])
def home():
    results = None
    dataset_used = None
    if request.method == 'POST':
        model_name = request.form["model"]
        dataset_type = request.form["dataset_type"]
        model = load_model(model_name)
        if dataset_type == 'builtin':
            dataset_name = request.form["dataset"]
            print(f"Selected dataset: {dataset_name}")
            X, y = load_dataset(dataset_name)
            dataset_used = dataset_name.upper()
        else:
            file = request.files['csv_file']
            df =pd.read_csv(file)
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            dataset_used = 'Uploaded CSV'
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
        scores = [acc, prec, rec, f1]
        plt.figure()
        plt.bar(metrics, scores, color=['blue', 'orange', 'green', 'red'])
        plt.ylim(0, 1)
        plt.title(f'Model Performance on {dataset_used}')
        plt.ylabel('Score')
        plt.savefig('static/metrics.png')
        plt.close()
        results = {
            'accuracy': format(acc, '.2f'),
            'precision': format(prec, '.2f'),
            'recall': format(rec, '.2f'),
            'f1': format(f1, '.2f'),
            "dataset": dataset_used
        }
    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)