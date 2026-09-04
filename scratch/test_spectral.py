import sys
sys.path.insert(0, r'd:\VIT\SIH 26\SIH26167\SatQuery-ISIH')
import numpy as np
from models.spectral_index_model import SpectralIndexModel

m = SpectralIndexModel()

# Create a test image with green and blue regions
img = np.zeros((256, 256, 3), dtype=np.uint8)
# Green region (vegetation)
img[0:128, 0:128] = [50, 180, 50]
# Blue region (water)
img[128:256, 128:256] = [30, 80, 200]
# Brown region (bare land)
img[0:128, 128:256] = [160, 120, 80]
# Mixed
img[128:256, 0:128] = [100, 150, 100]

# Test water detection
result = m.predict(img, 'Is there any water in this image?', {'index_type': 'ndwi'})
print('=== Water Detection ===')
print('Answer:', result['output']['answer'])
print('Confidence:', result['confidence'])
print('Detections:', len(result['output']['detections']))
print('Has annotated image:', result['output']['annotated_image'] is not None)
print()

# Test vegetation detection
result2 = m.predict(img, 'Detect vegetation in this satellite image', {'index_type': 'ndvi'})
print('=== Vegetation Detection ===')
print('Answer:', result2['output']['answer'])
print('Confidence:', result2['confidence'])
print('Detections:', len(result2['output']['detections']))
print('Has annotated image:', result2['output']['annotated_image'] is not None)
