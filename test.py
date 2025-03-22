import numpy as np

contours = [
    np.array([[[1, 2]], [[3, 4]]]),  # (2,1,2)
    np.array([[[1, 3]]])  # (1,1,2)
]

for c in contours:
    for point in c.reshape(-1, 2):
        print(point)
