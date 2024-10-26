import os
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import uvicorn
from pdf2image import convert_from_path
from app.inference import load_model, predict

# Create the 'temp' directory if it does not exist
os.makedirs('./temp', exist_ok=True)

app = FastAPI()
model = load_model()

def read_image(file_path):
    """Open an image file and convert it to RGB format."""
    try:
        image = Image.open(file_path).convert('RGB')
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not open image file: {e}")

def read_pdf(file_path):
    """Convert a single-page PDF to an image."""
    try:
        # Convert the first page of the PDF to an image
        images = convert_from_path(file_path, first_page=0, last_page=1, fmt="png")
        
        if len(images) != 1:
            raise HTTPException(status_code=400, detail="PDF must be a single-page document.")
        
        # Save the image as a temporary file
        image_path = f"./temp/{os.path.splitext(os.path.basename(file_path))[0]}.png"
        images[0].save(image_path, "PNG")
        return image_path
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not convert PDF to image: {e}")

def save_uploaded_file(upload_file: UploadFile):
    """Save the uploaded file to the 'temp' directory."""
    try:
        file_path = f"./temp/{upload_file.filename}"
        with open(file_path, "wb") as buffer:
            buffer.write(upload_file.file.read())
        return file_path
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save uploaded file: {e}")

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    """Handle image prediction requests."""
    # Save the uploaded file
    file_path = save_uploaded_file(file)

    # Determine the file type and process accordingly
    if file.filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        image = read_image(file_path)
        predicted_class, confidence = predict(image, model)
    elif file.filename.lower().endswith(".pdf"):
        # Convert PDF to an image
        image_path = read_pdf(file_path)
        image = read_image(image_path)
        predicted_class, confidence = predict(image, model)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload an image or a single-page PDF.")
    
    return {"class": predicted_class, "confidence": confidence}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
