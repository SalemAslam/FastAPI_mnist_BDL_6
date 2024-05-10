import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from io import BytesIO
from PIL import Image
import numpy as np
import uvicorn  # Server on which FastAPI runs
# "uvicorn main:app --reload" on terminal to view the web interface of FastAPI

# Importing helper functions
from load_model import load_model
from predict_format_image import predict_image, format_image

# Create the FastAPI application
app = FastAPI(title="Digit Recognition API (BE20B027)")


# Load the pre-trained MNIST model
model_path = "/Users/Salem Aslam/Documents/3. Academics/#Sem8/Lab/A6/MNIST_model.keras"
model = load_model(model_path)
# Set the model to inference mode for prediction
model.trainable = False


### Tasks 1 & 2

@app.post('/predict')
async def predict(uploaded_image: UploadFile = File(...)):
    """
    Predicts the digit present in an uploaded image.

    This endpoint accepts an image file as input and performs digit classification using 
    the pre-trained MNIST model. It first validates the image format (JPEG, JPG, or PNG), 
    then preprocesses the image by converting it to grayscale and resizing it to the model's 
    expected dimensions. Finally, the image is flattened and normalized before feeding it 
    to the model for prediction. The predicted digit label is returned in the response.

    Raises:
        HTTPException: If the uploaded file format is not supported.
    """

    # Read the image content from the uploaded file
    image_content = await uploaded_image.read()

    # Define accepted image formats
    accepted_formats = ['.jpeg', '.jpg', '.png']
    file_extension = os.path.splitext(uploaded_image.filename)[1].lower()

    # Validate image format
    if file_extension not in accepted_formats:
        raise HTTPException(status_code=400, detail="Bad file format. Accepted formats are .jpeg, .jpg, .png")

    # Extract filename without extension
    filename = os.path.splitext(uploaded_image.filename)[0]

    # Open the image from the byte stream
    image = Image.open(BytesIO(image_content))

    # Convert the image to grayscale
    grayscale_image = image.convert('L')

    # Preprocess the image using the helper function
    preprocessed_image = format_image(grayscale_image)

    # Predict the digit using the loaded model and helper function
    digit = predict_image(model, preprocessed_image)

    # Return the predicted digit in the response
    return {"digit": digit}

