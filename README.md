NSFW Content Filter

Introduction

Welcome to the NSFW content filter, contains a FastAPI application for detecting NSFW (Not Safe for Work) content in images. The application is designed to help, small social media companies or other platforms, filter out potentially inappropriate or offensive content from user-uploaded images.

Features

Image Upload: Users can upload one or more images to be analyzed for NSFW content.

NSFW Content Detection: The uploaded images are analyzed using a pre-trained deep learning model. The model has been fine-tuned to detect NSFW content with high accuracy.

Model Details

The NSFW content detection model used in this application has been trained to achieve a high level of accuracy:

Training Accuracy: 98.5%
Validation Accuracy: 98.6%
This level of accuracy ensures that the model can effectively identify NSFW content in images.

Usage

To run this code, you can use Uvicorn with the following command:

```
uvicorn app:app --host 127.0.0.1 --port 8001
```

Once the server is up and running, you can access the API documentation and test the NSFW content filter in your browser at http://localhost:8001/docs. The API provides an endpoint for uploading images, and it returns a list of boolean values indicating whether each uploaded image passes the NSFW filter or not.