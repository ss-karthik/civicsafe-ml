import io
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for testing only)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (POST, GET, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Load models once at startup to save time
model_paths = ['dividers_best.pt', 'garbage_best.pt', 'potholes_best.pt', 'streetlight_best.pt', 'wires_best.pt']
models = [YOLO(path) for path in model_paths]

@app.post("/detect-classes")
async def detect_classes(file: UploadFile = File(...)):
    # 1. Read image from the upload
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # 2. Convert to format YOLO understands (numpy array)
    img_np = np.array(img)
    
    detected_classes = set()
    
    # 3. Run inference across all loaded models
    for model in models:
        results = model(img_np, conf=0.4, verbose=False)
        for r in results:
            for c in r.boxes.cls:
                class_name = model.names[int(c)]
                detected_classes.add(class_name)

    result = set()
    for cls in detected_classes:
        if cls=="broken":
            result.add("broken_divider")
        elif cls=="Not Working":
            result.add("broken_streetlight")
        elif cls=="tangled_broken_wires":
            result.add("tangled_wires")
        else:
            result.add(cls)
        
    return {"detected_classes": list(result)}

@app.get("/")
async def root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



