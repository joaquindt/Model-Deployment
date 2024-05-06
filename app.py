import os
from flask import Flask, request, jsonify, render_template, redirect, url_for
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename


app = Flask(__name__)

# Dummy data loading - replace or adjust this as needed.
# For deployment, ensure the data is accessible in the environment where your app is hosted.
#df = pd.read_excel('quantitative_data.xlsx')  # Ensure the file is in the correct directory on PythonAnywhere.

UPLOAD_FOLDER = 'uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def ahp(responses):
    # Assuming responses is a DataFrame similar to df
    responses = responses.dropna()
    aux1 = {i: 16-i for i in range(1, 17)}
    for index, row in responses.iterrows():
        for col in responses.columns:
            responses.at[index, col] = aux1[row[col]]
    
    average_array = [responses[col].mean() for col in responses.columns]
    
    comparison_matrix = np.zeros((16, 16))
    n = 16
    for i in range(n):
        for j in range(n):
            if i != j:
                comparison_matrix[i, j] = average_array[i] / average_array[j]
            else:
                comparison_matrix[i, j] = 1
    comparison_matrix = pd.DataFrame(comparison_matrix)
    
    column_sums = np.sum(comparison_matrix, axis=0)
    normalized_matrix = comparison_matrix / column_sums
    priority_vector = np.mean(normalized_matrix, axis=1)
    
    weighted_sum = np.dot(comparison_matrix, priority_vector)
    lambda_max = np.sum(weighted_sum / priority_vector) / n
    CI = (lambda_max - n) / (n - 1)
    RI = 1.61  # Assuming RI for n=16 is approximately 1.61
    CR = CI / RI
    
    max_weighing = priority_vector.max()
    grades = (priority_vector * 10) / max_weighing
    final = pd.DataFrame({'Criteria': responses.columns, 'Weight': priority_vector, 'Grade': grades})
    final = final.sort_values(by='Weight', ascending=False)
    
    return final.to_dict(orient='records')


@app.route('/', methods=['GET'])
def upload_file():
    return render_template('upload.html')

@app.route('/uploader', methods=['POST'])
def upload_file_post():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)  # Save the file
        df = pd.read_excel(filepath)  # Process the file
        results = ahp(df)  # Call your AHP function
        return render_template('results.html', results=results)  # Display results
    return 'File type not allowed'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'xlsx', 'xls'}

if __name__ == '__main__':
    app.run(debug=True, port=8080)