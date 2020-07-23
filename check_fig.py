import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib


original_image = plt.imread("me.jpg") # values between 0 and 255 it seems
fig, ax = plt.subplots()
image = original_image.copy() / 255

ax.imshow(image, aspect='equal')

fig.savefig('check_fig.jpg')
