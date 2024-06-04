
from flask import Flask, request, render_template, jsonify, send_from_directory
import pandas as pd
import os
from prophet import Prophet
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

if not os.path.exists(app.config['STATIC_FOLDER']):
    os.makedirs(app.config['STATIC_FOLDER'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        try:
            data = pd.read_csv(filepath)
            columns = data.columns.tolist()
            return jsonify({"columns": columns, "filename": file.filename}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    filename = request.form['filename']
    target_column = request.form['target_column']
    n_periods = int(request.form['n_periods'])
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        data = pd.read_csv(filepath)
        if target_column not in data.columns:
            return jsonify({"error": "Column not found"}), 400
        col_ds = data.columns[0]
        data.rename(columns={target_column: 'y',col_ds:'ds'}, inplace=True)
        data['ds'] = pd.to_datetime(data['ds'])
        forecast = make_prediction(data, n_periods)
        image_path = save_plot(data,forecast)
        return jsonify({"message": "Prediction completed", "image_path": image_path}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def make_prediction(df, n_periods):
    """
    :param n_periods: Number of periods to forecast
    :param df: Original data
    :return: Forecasted data
    """
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=n_periods)
    forecast = m.predict(future)
    return forecast

def save_plot(df,forecast):
    plt.figure(figsize=(10, 6))
    plt.plot(df['ds'], df['y'], label='True')
    plt.plot(forecast['ds'], forecast['yhat'], label='Forecast')
    plt.legend()
    image_path = os.path.join(app.config['STATIC_FOLDER'], 'forecast.png')
    plt.savefig(image_path)
    plt.close()
    return image_path

if __name__ == '__main__':
    app.run(debug=True)









