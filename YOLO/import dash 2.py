import dash
from dash import dcc, html, Input, Output, State
import dash_uploader as du
import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import plotly.express as px
import base64
from io import BytesIO
from PIL import Image

# Load the trained YOLO model
model_path = ("runs/detect/train/weights/best.pt")
model = YOLO(model_path)  # Load the model directly from the saved weights
print("✅ Model loaded successfully!")

# Create upload folder
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("PCB Defect Detection", style={'textAlign': 'center'}),
    
    # Upload Image
    dcc.Upload(
        id="upload-image",
        children=html.Button("Upload Image", style={"fontSize": 20}),
        multiple=False
    ),

    # Display uploaded image
    html.Div(id="output-image-upload"),

    # Process & Detect button
    html.Button("Detect Defects", id="detect-button", n_clicks=0, style={"fontSize": 20, "marginTop": "20px"}),

    # Display results
    html.Div(id="output-detection")
])

# Callback to process uploaded image
@app.callback(
    Output("output-image-upload", "children"),
    Input("upload-image", "contents"),
    State("upload-image", "filename")
)
def display_uploaded_image(contents, filename):
    if contents is not None:
        # Convert base64 string to image
        _, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        img_path = os.path.join(UPLOAD_FOLDER, filename)

        # Save the uploaded image
        with open(img_path, "wb") as f:
            f.write(decoded)

        return html.Img(src=contents, style={"width": "50%", "marginTop": "20px"})

    return None

# Callback to run YOLO model
@app.callback(
    Output("output-detection", "children"),
    Input("detect-button", "n_clicks"),
    State("upload-image", "filename")
)
def detect_defects(n_clicks, filename):
    if n_clicks > 0 and filename is not None:
        img_path = os.path.join(UPLOAD_FOLDER, filename)

        # Run YOLO detection
        results = model(img_path)

        # Process result image
        img = results[0].plot()  # Draw bounding boxes on the first result
        output_path = os.path.join(UPLOAD_FOLDER, "detected_" + filename)
        cv2.imwrite(output_path, img)

        # Convert image to base64 for display
        _, buffer = cv2.imencode('.jpg', img)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        img_src = f"data:image/jpeg;base64,{encoded_image}"

        return html.Img(src=img_src, style={"width": "50%", "marginTop": "20px"})

    return None

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
