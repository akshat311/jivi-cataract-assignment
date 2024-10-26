# Cataract Classification API <br>(Akshat Agarwal)
This project provides a machine learning model and an API for binary classification of eye images, determining whether an image shows a cataract or a normal eye. The API is built using FastAPI and accepts image files (JPEG, PNG, BMP) or single-page PDFs as input.

## Project Overview
The Cataract Classification API provides a simple interface for uploading an image or a single-page PDF and predicting whether the image represents a cataract or a normal eye. The API returns a classification result and a confidence score, with support for different image formats and PDFs.

## Environment Setup
Clone the Repository:

```bash
git clone <repository-url>
cd cataract_classification
```
Set Up a Virtual Environment (recommended):

```bash
python3 -m venv env
source env/bin/activate
```
Install Required Dependencies: Install the required Python libraries from requirements.txt:

```bash
pip install -r requirements.txt
```
Install Poppler for PDF Support (required by pdf2image):

On Ubuntu:
```bash
sudo apt-get install poppler-utils
```
On Mac (using Homebrew):
```bash
brew install poppler
```
## Dataset Setup
Organize the Dataset: Place the dataset images in the following folder structure under data/processed_images:


```
data/
└── processed_images/
    ├── train/
    │   ├── cataract/
    │   └── normal/
    └── test/
        ├── cataract/
        └── normal/
```
The train and test folders should contain subfolders for each class, cataract and normal, with images for each class in the respective folders.


## Model Training
Configure Training Parameters: You can adjust training parameters (e.g., batch size, number of epochs) in train.py.

Train the Model: Run the following command to train the model:

```bash

python train.py
```
This will:

Load and preprocess the dataset.
Train a pre-trained model on the data.
Save the trained model weights to models/cataract_classification_model.pth.
Evaluation: To evaluate the trained model, run:

```bash
python evaluate.py
```
This will output metrics such as accuracy, classification report, confusion matrix, and an ROC curve.

## API Deployment
Start the FastAPI Server: Run the following command to deploy the API locally:

```bash
uvicorn server:app --host localhost --port 8000
```
This will start the FastAPI server on http://localhost:8000.
API Documentation:

Open a browser and navigate to http://localhost:8000/docs to view the interactive API documentation provided by FastAPI.
API Documentation
The Cataract Classification API provides an endpoint for uploading an image or a single-page PDF to classify whether it contains a cataract or a normal eye.

Base URL
```
http://localhost:8000
```
### Endpoint: `/predict/`

* Description: Accepts an image or single-page PDF file, processes it, and returns the predicted class (cataract or normal) with a confidence score.
* Method: `POST`
* Content Type: multipart/form-data

### Request Parameters
* file (required): The image file or single-page PDF to be uploaded.
* Supported image formats: .jpg, .jpeg, .png, .bmp
* Supported PDF format: Single-page .pdf only
### Example Request
Using curl:

```bash
curl -X POST "http://localhost:8000/predict/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/image_or_pdf_file.jpg"
```

### Using Python requests:

```python
import requests

url = "http://localhost:8000/predict/"
file_path = "/path/to/image_or_pdf_file.jpg"

with open(file_path, "rb") as file:
    response = requests.post(url, files={"file": file})

print(response.json())
```

### Example Response
```json

{
  "class": "cataract",
  "confidence": 0.87
}
```
### Response Parameters
class: The predicted class (cataract or normal).
confidence: The confidence score of the prediction, ranging from 0 to 1.
Error Handling
The API includes robust error handling with informative messages for common issues. Below are possible error responses and their causes:

### 400 Bad Request:

The uploaded file format is unsupported (neither an image nor a single-page PDF).
The PDF contains more than one page.
The PDF cannot be converted to an image.
The image file cannot be read.
Example Error Response for Unsupported File Format:

```json

{
  "detail": "Unsupported file format. Please upload an image or a single-page PDF."
}
```

### Example Error Response for Multi-Page PDF:

```json
{
  "detail": "PDF must be a single-page document."
}
```

### 500 Internal Server Error:

The server encountered an unexpected issue that prevented it from fulfilling the request, such as a file-saving error.
Example Error Response for File Saving Issue:

```json
{
  "detail": "Could not save uploaded file."
}
```

## Dependencies
The project requires the following dependencies:

Python Libraries:
```
fastapi
uvicorn
torch
pdf2image
Pillow
PyPDF2
scikit-learn
matplotlib
```

Poppler for PDF Support:

Install using 
```sudo apt-get install```
poppler-utils on Ubuntu or brew install poppler on Mac.
## Additional Notes
* PDF Support: The API supports only single-page PDFs. Multi-page PDFs will return a 400 Bad Request error.
* File Handling: Ensure that uploaded files are valid image formats or a single-page PDF. Invalid files will result in an error message.
* File Cleanup: Uploaded files are stored temporarily in a temp directory. In production, ensure that files are cleaned up periodically to save disk space.