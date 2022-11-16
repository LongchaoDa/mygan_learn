import math
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Assign main directory to a variable
main_dir = os.path.dirname(sys.path[0])


# A function to get coordinates of points on the circle's circumference
def PointsInCircum(r, n=100):
    return [(math.cos(2 * math.pi / n * x) * r, math.sin(2 * math.pi / n * x) * r) for x in range(0, n + 1)]


# Save coordinates of a set of real points making up a circle with radius=2
circle = np.array(PointsInCircum(r=2, n=1000))

# Draw a chart
plt.figure(figsize=(15, 15), dpi=400)
plt.title(label='Real circle to be learned by the GAN generator', loc='center')
plt.scatter(circle[:, 0], circle[:, 1], s=5, color='black')
plt.show()
