import numpy as np


# Random data in [0, 0.8]^d
def uniform(nbr_elements, dimension):
    data = np.array([np.random.random(dimension)*0.8 for i in range(nbr_elements)])
    return data

#


# Random distribution following a fractal shape.
# Has disjointed elements.
# Only works in 2D for now
def sierpinski_carpet(nbr_elements, dimension=2, level=2):
    data = np.array([np.random.random(2) for i in range(nbr_elements)])
    for i in range(nbr_elements):
        correct = False
        while not correct:
            j = 0
            data[i] = np.random.random(2)
            x = data[i][0]
            y = data[i][1]
            while j < level and not correct:
                j += 1
                if 1/3 < x < 2/3 and 1/3 < y < 2/3:
                    correct = True
                else:
                    x = (x % (1/3)) * 3
                    y = (y % (1/3)) * 3
    return data
