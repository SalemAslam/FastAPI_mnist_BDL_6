## FastAPI MNIST Digit Prediction

### Introduction

This project implements a FastAPI application functioning as a digit recognition API. Users can interact with the API by uploading an image containing a handwritten digit, and the API predicts the digit using a pre-trained MNIST model.

### About the Assignment

This project utilizes Python libraries like FastAPI, Pillow (PIL Fork), NumPy, and TensorFlow (through Keras) to achieve its functionality. Core functionalities include:

- **API Endpoint (/predict):**
  - Accepts image uploads via POST requests.
  - Validates image format (JPEG, JPG, or PNG).
  - Preprocesses the image by converting to grayscale and resizing to the model's expected dimensions (28x28).
  - Flattens and normalizes the image data for model compatibility.
  - Performs prediction using a pre-trained MNIST model, returning the predicted digit label in the API response.

- **Helper Functions:**
  - *load_model:* Loads the pre-trained MNIST model from a specified file path.
  - *predict_image:* Takes a model and a preprocessed image, performs prediction, and returns the predicted digit class as a string.

### Model Architecture

The neural network architecture for digit recognition consists of:
- Input layer for flattened images (784 pixels).
- Two hidden layers with 256 & 128 neurons (sigmoid activation).
- Output layer with 10 neurons (softmax activation for digit probabilities).

### Results

The model achieved accurate predictions on the original MNIST test data, correctly identifying all 10 digits. However, its performance was poor when presented with custom handwritten digits from MS Paint, incorrectly predicting most digits except for '1'. 

The discrepancy may arise from differences between the training data and custom inputs, suggesting the model's inability to generalize to unseen variations in handwritten digit appearances. Augmenting the training data with diverse styles could enhance the model's performance.

### Conclusion

While the model performs well on the original MNIST data, its reliance on unaugmented training data limits its robustness to variations in handwritten digit styles. Implementing data augmentation techniques during training could improve the model's ability to handle diverse digit appearances. 

GitHub Repository: [FastAPI MNIST Digit Prediction](https://github.com/SalemAslam/FastAPI_mnist_BDL_6)
