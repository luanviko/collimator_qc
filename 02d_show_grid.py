import matplotlib.pyplot as plt
from PIL import Image

img = Image.open('natural_grid.jpg')
plt.imshow(img, origin='upper')
plt.axis('on') 
plt.show()