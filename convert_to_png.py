from PIL import Image

# Open the PPM file
im = Image.open('cmake-build-release/output.ppm')

# Save as PNG
im.save('output.png')
