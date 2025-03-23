AI for Star & Exoplanet Detection - Deployable Project
🚀 Project Overview
This project aims to automatically detect and track celestial objects from NASA's NEOWISE & Hubble telescope images using AI & Deep Learning.
🛠️ Tech Stack
• Python
• TensorFlow / PyTorch
• OpenCV (Image processing)
• Astropy (FITS image handling)
• Google Cloud / AWS (For deployment)
• YOLOv8 (Object detection)
• FastAPI (For API-based deployment)
🔹 Step 1: Setup & Install Dependencies
Install the required libraries using the following command:
```bash
pip install numpy pandas matplotlib opencv-python-headless tensorflow keras ultralytics torch torchvision torchaudio fastapi uvicorn astropy astroquery
```
🔹 Step 2: Fetch NASA Telescope Images
Use Astroquery to fetch images from NASA's NEOWISE archive.
```python
from astroquery.irsa import Irsa
from astropy.coordinates import SkyCoord
import astropy.units as u

Irsa.ROW_LIMIT = 10
coord = SkyCoord('08h52m00.0s +18d00m00s', unit=(u.hourangle, u.deg))
table = Irsa.query_region(coord, catalog='allwise_p3as_mep', spatial='Cone', radius=0.5 * u.deg)
print(table)
```
🔹 Step 3: Load & Display FITS Images
Load and visualize the FITS images from NASA.
```python
from astropy.io import fits
import matplotlib.pyplot as plt

fits_file = 'nasa_image.fits'
hdulist = fits.open(fits_file)
image_data = hdulist[0].data  
plt.figure(figsize=(10, 10))
plt.imshow(image_data, cmap='gray', origin='lower')
plt.colorbar(label='Pixel Intensity')
plt.title('NASA Telescope Image')
plt.show()
```
🔹 Step 4: Image Preprocessing (Denoising & Contrast Enhancement)
Apply Gaussian blur and normalize contrast.
```python
import cv2
import numpy as np

blurred_image = cv2.GaussianBlur(image_data, (5, 5), 0)
norm_image = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))
plt.imshow(norm_image, cmap='gray', origin='lower')
plt.title('Preprocessed NASA Image')
plt.show()
```
🔹 Step 5: AI Model for Detecting Stars & Exoplanets
Train YOLOv8 on NASA images.
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='nasa_dataset.yaml', epochs=50, imgsz=640)
```
🔹 Step 6: Detect Moving Objects (Potential Exoplanets)
Compare images taken at different times to detect motion.
```python
image1 = fits.getdata('image_1.fits')
image2 = fits.getdata('image_2.fits')
diff_image = np.abs(image2 - image1)
plt.imshow(diff_image, cmap='gray', origin='lower')
plt.colorbar(label='Difference Intensity')
plt.title('Moving Object Detection')
plt.show()
```
🔹 Step 7: Deploy the AI Model with FastAPI
Create `app.py` and deploy the model as an API.
```python
from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
from ultralytics import YOLO

app = FastAPI()
model = YOLO('yolov8n.pt')

@app.post('/detect/')
async def detect_objects(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    results = model(img)
    return {'detections': results.pandas().xyxy[0].to_dict(orient='records')}
```
🔹 Step 8: Deploy the Model to Cloud
Deploy the AI model to Google Cloud or AWS.
```bash
gcloud ai custom-jobs create --region=us-central1 --display-name='planet-detection' --python-package-uris=gs://my_bucket/yolo_model.tar.gz --python-module=my_model.main --machine-type=n1-standard-4
```
🔹 Step 9: Automating Continuous Monitoring
Schedule an automatic script to fetch and analyze images.
```python
import schedule
import time

def fetch_and_analyze():
    images = Irsa.query_region(SkyCoord('08h52m00.0s +18d00m00s'), catalog='allwise_p3as_mep', radius=0.5 * u.deg)
    for image in images:
        results = model(image)
        print('Detected objects:', results)

schedule.every().day.at('00:00').do(fetch_and_analyze)
while True:
    schedule.run_pending()
    time.sleep(1)
```
🌍 Final Summary: What You Get
✅ End-to-End AI Model for detecting exoplanets
✅ Real-time Object Detection API using FastAPI
✅ Cloud Deployment on Google Cloud / AWS
✅ Automated Image Processing Pipeline
