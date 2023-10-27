import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import shutil
import base64
from typing import List
from io import BytesIO
from typing import List, Dict
from pydantic import BaseModel
from PIL import Image
import sys
import os
from os.path import isfile, join
import torch
import torchvision
from torchvision import datasets, models, transforms
import uuid

#Setup
app = FastAPI(max_request_size=500 * 1024 * 1024) # 500MB-limit

#Cors Setup
origins = [
    # Add allowed origins
    "https://localhost",
    "https://localhost:3000",
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Static Files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/")
async def Detector(images: List[bytes] = File(...)):
    results = []

    for img_data in images:
        image_name = uuid.uuid4().hex
        image_path = f"./static/images/{image_name}.jpg"
        with open(image_path, "wb") as f:
            f.write(img_data)

        # Load Model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = torch.load('./static/models/ptm=resnet50_batch=16_step-size=14_lr=0.006.pth', map_location=device).to(device)
        model.eval()

        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.25, 0.25, 0.25])
        transform_tensor = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

        img = Image.open(image_path).convert('RGB')
        img_y = transform_tensor(img).unsqueeze(0).float().to(device)

        prediction = torch.argmax(model(img_y))
        if prediction == 0:
            results.append(True)
        else:
            results.append(False)

        # Clean up: Remove the image file
        if os.path.exists(image_path):
            os.remove(image_path)

    return results
    
if __name__ == '__main__':
    uvicorn.run(app,host="127.0.0.1",port="8001")