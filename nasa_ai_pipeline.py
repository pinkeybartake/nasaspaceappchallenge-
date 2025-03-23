import os
import requests
import json
import re
import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from fastapi import FastAPI, Query
from astroquery.skyview import SkyView
from astropy.io import fits
from ultralytics import YOLO

# Initialize FastAPI app
app = FastAPI()

# NASA Horizons API Configuration
HORIZONS_API_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"

PLANET_IDS = {
    "Mercury": "199",
    "Venus": "299",
    "Earth": "399",
    "Mars": "499",
    "Jupiter": "599",
    "Saturn": "699",
    "Uranus": "799",
    "Neptune": "899",
    "Pluto": "134340",
    "Moon": "301",
    "ISS": "25544"
}

import requests
import json
import re
from datetime import datetime, timedelta

import requests
import json
import re
from datetime import datetime, timedelta

import requests
from datetime import datetime, timedelta

import requests
from datetime import datetime, timedelta

import requests
from datetime import datetime, timedelta

import requests
from datetime import datetime, timedelta

import requests
import re
from datetime import datetime, timedelta

import requests
import re
from datetime import datetime, timedelta

import requests
import re
from datetime import datetime, timedelta

import requests
from datetime import datetime, timedelta

def get_real_time_coordinates(planet_name):
    print(f"🔍 Fetching real-time position for {planet_name}...")

    # NASA Horizons API Endpoint
    url = "https://ssd.jpl.nasa.gov/api/horizons.api"

    # Define correct IDs for planets, Moon, and ISS
    planet_ids = {
        "Mercury": "199",
        "Venus": "299",
        "Earth": "399",
        "Mars": "499",
        "Jupiter": "599",
        "Saturn": "699",
        "Uranus": "799",
        "Neptune": "899",
        "Pluto": "134340",
        "Moon": "301",
        "ISS": "25544"
    }

    # Get correct Horizons COMMAND ID
    command = planet_ids.get(planet_name.capitalize(), f"'{planet_name}'")

    # Get current date in YYYY-MMM-DD HH:MN format
    now = datetime.utcnow()
    start_time = now.strftime('%Y-%b-%d %H:%M')  # Correct date format
    stop_time = (now + timedelta(minutes=1)).strftime('%Y-%b-%d %H:%M')  # 1 min later

    # Fetch RA and DEC using QUANTITIES=1
    def fetch_coordinates():
        params = {
            "format": "json",
            "COMMAND": command,
            "EPHEM_TYPE": "OBSERVER",
            "CENTER": "500@399",
            "START_TIME": f'"{start_time}"',
            "STOP_TIME": f'"{stop_time}"',
            "STEP_SIZE": "1m",
            "QUANTITIES": "1"  # ✅ Use '1' to get both RA and DEC
        }
        try:
            response = requests.get(url, params=params)
            if response.status_code != 200:
                print(f"❌ Error: Received HTTP {response.status_code}")
                return None

            data = response.json()
            if "result" in data:
                lines = data["result"].split("\n")
                extract = False
                for line in lines:
                    if "$$SOE" in line:
                        extract = True
                        continue
                    if "$$EOE" in line:
                        break
                    if extract and len(line.strip()) > 0:
                        values = line.split()
                        if len(values) >= 8:
                            ra = f"{values[2]}h {values[3]}m {values[4]}s"
                            dec = f"{values[5]}° {values[6]}' {values[7]}\""
                            return ra, dec

            print(f"❌ No valid RA/DEC data found for {planet_name}.")
            return None, None

        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching coordinates for {planet_name}: {e}")
            return None, None

    ra, dec = fetch_coordinates()

    if ra and dec:
        ra_dec_str = f"{ra} {dec}"
        print(f"✅ Real-Time Position of {planet_name}: {ra_dec_str}")
        return ra_dec_str

    print(f"❌ Failed to retrieve valid coordinates for {planet_name}.")
    return None




# Example usage
#get_real_time_coordinates("Mars")

#



# Step 1: Fetch & Download NASA FITS Images
def fetch_nasa_fits(target="Pluto", survey="WISE 12"):
    print(f"🔍 Fetching NASA image for {target} from {survey}...")

    # Get real-time coordinates
    coords = get_real_time_coordinates(target)
    if coords is None:
        print("❌ Unable to fetch coordinates. Using default target name.")
        coords = target  # Fall back to planet name

    try:
        # Pass properly formatted coordinates to SkyView
        image_urls = SkyView.get_image_list(position=coords, survey=survey)

        if not image_urls:
            raise ValueError("No images found for the given target.")

        fits_filename = "nasa_image.fits"

        # Download the first available image
        response = requests.get(image_urls[0])
        with open(fits_filename, 'wb') as file:
            file.write(response.content)

        print(f"✅ Downloaded NASA FITS image: {fits_filename}")
        return fits_filename

    except Exception as e:
        print(f"❌ Error fetching NASA images: {e}")
        return None


# Step 2: Process FITS Image
def process_fits_image(fits_file):
    print("🛠 Processing FITS image...")
    
    try:
        hdulist = fits.open(fits_file)
        image_data = hdulist[0].data

        # Normalize contrast
        norm_image = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))

        # Apply Gaussian Blur to remove noise
        blurred_image = cv2.GaussianBlur(norm_image, (5, 5), 0)

        # Save processed image
        processed_filename = "processed_image.png"
        plt.imsave(processed_filename, blurred_image, cmap="gray")

        print(f"✅ Processed image saved as {processed_filename}")
        return processed_filename
    except Exception as e:
        print(f"❌ Error processing FITS image: {e}")
        return None

# Step 3: AI Model to Detect Celestial Objects (YOLOv8)
def detect_objects(image_path):
    print("🚀 Running AI model for detection...")
    
    try:
        model = YOLO("yolov8n.pt")
        results = model(image_path)

        results_img = "detections.png"
        results[0].save(results_img)

        print(f"✅ AI detection complete. Results saved in {results_img}")
        return results_img
    except Exception as e:
        print(f"❌ Error in AI detection: {e}")
        return None



# Step 4: Deploy as an API
@app.get("/")
def read_root():
    return {"message": "Welcome to NASA AI Image Processing API"}

@app.get("/planet_position")
def get_planet_position(planet: str = "Pluto"):
    coords = get_real_time_coordinates(planet)
    return {"planet": planet, "real_time_coordinates": coords} if coords else {"error": "Unable to fetch coordinates"}

@app.get("/fetch")
def fetch_image(target: str = Query("Pluto"), survey: str = Query("WISE 12")):
    fits_file = fetch_nasa_fits(target, survey)
    return {"message": f"Image fetched and saved as {fits_file}"} if fits_file else {"error": "Could not fetch image"}

@app.get("/process")
def process_image():
    processed_file = process_fits_image("nasa_image.fits")
    return {"message": f"Image processed and saved as {processed_file}"} if processed_file else {"error": "Could not process image"}

@app.get("/detect")
def detect_objects_api():
    detection_file = detect_objects("processed_image.png")
    return {"message": f"Detection complete, results saved in {detection_file}"} if detection_file else {"error": "Could not run AI detection"}
