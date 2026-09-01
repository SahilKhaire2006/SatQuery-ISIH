#!/usr/bin/env python3
"""
Create a test satellite image for testing SatQuery
"""

from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Create a realistic-looking satellite image
width, height = 512, 512

# Create base image with greenish-brown terrain
img = Image.new('RGB', (width, height), color=(120, 140, 100))
draw = ImageDraw.Draw(img)

# Add some "urban" areas (darker gray rectangles - buildings)
buildings = [
    (50, 50, 100, 80),
    (120, 60, 160, 90),
    (200, 100, 250, 150),
    (300, 150, 350, 200),
    (400, 200, 450, 240),
    (100, 250, 140, 290),
    (250, 300, 300, 350),
]

for building in buildings:
    draw.rectangle(building, fill=(80, 80, 90))

# Add some "roads" (lighter lines)
roads = [
    [(0, 150), (512, 150)],  # Horizontal road
    [(0, 300), (512, 300)],  # Another horizontal road
    [(200, 0), (200, 512)],  # Vertical road
    [(350, 0), (350, 512)],  # Another vertical road
]

for road in roads:
    draw.line(road, fill=(150, 150, 140), width=8)

# Add some "water bodies" (blue areas)
draw.ellipse([320, 350, 420, 450], fill=(70, 120, 180))
draw.ellipse([30, 350, 90, 410], fill=(60, 110, 170))

# Add some "vegetation" patches (darker green)
draw.ellipse([150, 400, 230, 480], fill=(70, 130, 60))
draw.ellipse([260, 50, 310, 90], fill=(60, 120, 50))

# Add a label
try:
    font = ImageFont.truetype("arial.ttf", 16)
except:
    font = ImageFont.load_default()

draw.text((10, 10), "Test Satellite Image", fill=(255, 255, 255), font=font)

# Save the image
output_path = 'data/raw/test_satellite_image.png'
img.save(output_path)
print(f"✓ Test image created: {output_path}")
print(f"  Size: {width}x{height}")
print(f"  Contains: buildings, roads, water bodies, vegetation")
print(f"\nYou can now use this image to test the CLI!")
