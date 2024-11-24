from PIL import Image

# Open the PPM file
im = Image.open('output.ppm')

# Save as PNG
im.save('output.png')
