import os
from flask import Flask, request, jsonify
# from PIL import Image
# import io
import base64
import tensorflow as tf
# from tensorflow import keras
import numpy as np
import mysql.connector

app = Flask(__name__)

# Function to preprocess the image
def preprocess_image(image_base64):
    # Logic to preprocess the image
    # Example: preprocessed_image = ...
    # Return the preprocessed image
    return preprocess_image

# Function to load the model
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Fetch image from Cloud SQL and process for prediction
@app.route('/predict', methods=['POST'])
def predict_image():
    try:
        # Connect to the Cloud SQL database
        connection = mysql.connector.connect(
            host='34.101.196.0',
            port='3306',
            database='testapiv2',
            user='root',
            password='anwari123'
        )

        # Retrieve the latest image URL from the Cloud SQL database
        cursor = connection.cursor()
        query = "SELECT url FROM images ORDER BY id DESC LIMIT 1"
        cursor.execute(query)
        result = cursor.fetchone()

        if not result:
            return jsonify({'error': 'Image not found'})

        image_url = result[0]

        # Fetch the image data from Cloud Storage
        response = request.get(image_url)
        image_data = response.content

        # Convert the image data to a PIL Image object
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        # Preprocess the image
        preprocessed_image = preprocess_image(image_base64)

        # Load the model
        model = load_model('model.h5')

        # Perform prediction on the preprocessed image
        prediction = model.predict(np.array([preprocessed_image]))

        # Logic for prediction result
        # ...

        return jsonify({'prediction': prediction})

    except Exception as e:
        print('Error processing image:', e)
        return jsonify({'error': 'Image processing failed'})

if __name__ == "__main__":
    app.run(debug=False,
            host="0.0.0.0",
            port=int(os.environ.get("PORT", 8080)))