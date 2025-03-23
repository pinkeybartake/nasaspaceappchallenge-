from astroquery.ipac.irsa import Irsa
from astropy.coordinates import SkyCoord
import astropy.units as u
import requests

# Set the IRSA service
Irsa.ROW_LIMIT = 10  

# Define target coordinates (Example: Pluto)
coord = SkyCoord("08h52m00.0s +18d00m00s", unit=(u.hourangle, u.deg))

# Query NEOWISE images
table = Irsa.query_region(coord, catalog="allwise_p3as_mep", spatial="Cone", radius=0.5 * u.deg)

# Get the first image URL from NASA's dataset
image_url = table['download'].data[0]

# Download the image to Colab
fits_filename = "nasa_image.fits"
response = requests.get(image_url)

# Save the FITS file
with open(fits_filename, 'wb') as file:
    file.write(response.content)

print(f"✅ Downloaded NASA FITS image: {fits_filename}")
