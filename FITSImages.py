from astropy.io import fits
import matplotlib.pyplot as plt

# Load FITS file
fits_file = "nasa_image.fits"
hdulist = fits.open(fits_file)

# Extract image data
image_data = hdulist[0].data  

# Display image
plt.figure(figsize=(10, 10))
plt.imshow(image_data, cmap='gray', origin='lower')
plt.colorbar(label="Pixel Intensity")
plt.title("NASA Telescope Image")
plt.show()
