import numpy as np

#################### Simulation parameters ####################
second = 1.0
millisecond = 0.001
dt = 10 * millisecond
nb_steps = 100

####################   DNF parameters ####################
width = 100  # size of the field
height = 100  # size of the field
dnf_shape = (width, height)

tau = 0.92 * dt
# tau = 0.2*dt
h = -0.7
# Ap and Sp : Amplitude and sigma of positive gaussian
Ap = 4.0
Sp = 0.01
# Am and Sm : Amplitude and sigma of negative gaussian
Am = 2.0
Sm = Sp / 3

# global inhibition
gi = 50


# Version normalization discrete Gaussian
def gaussian(shape=(25, 25), sigma=0.5, center=0.5, normalized=True):
    norm_cst = 0.0
    res = np.zeros(shape, dtype=float)

    # Generate a gaussian of the form g(x) = height*exp(-(x-center)²/σ²). '''
    if type(shape) in [float, int]:  # if type is not a tuple, e.g. (25,25) but a float eg 25
        shape = (shape,)  # then cast shape as a tuple
    if type(sigma) in [float, int]:
        sigma = (sigma,) * len(
            shape)  # define sigma for a given nb of dimensions, e.g. by default sigma [0.5] becomes a tuple (0.5,0.5)
    if type(center) in [float, int]:
        center = (center,) * len(
            shape)  # define the center for a given nb of dimensions, e.g. by default center [0] becomes a tuple (0,0)
    grid = []
    for size in shape:
        grid.append(slice(0,
                          size))  # add slices to the grid list, by default it will insert two slices of (0,1,...,25) in the grid
    C = np.mgrid[tuple(
        grid)]  # creates a 3D meshgrid, filled with indices corresponding to shape : by default of shape (25,25) (25,25)
    R = np.zeros(shape, dtype=float)  # create an array filled of zeros (by default 25*25 array)
    for i, size in enumerate(
            shape):  # enumerate allows to get a list of tuples (indice, value) -> (i,size) in function of the content of the list shape (25,25)
        if shape[i] > 1:
            # compute squared normalized euclidian distance between the neurons (normalization : divide by size),
            #  for each dimension (xi-xj/Nx)**2 then (yi-yj/Ny)**2
            # and multiply the squared distance for each dimension by -1/2*sigma**2
            R += (((C[i] / float(size - 1)) - center[i]) / sigma[i]) ** 2
    if normalized:
        res = np.exp(-R / 2)
        for index, val in np.ndenumerate(res):
            norm_cst += val
        # print ("norm_cst =",norm_cst)
        return np.exp(-R / 2) / norm_cst
    else:
        return np.exp(-R / 2)


# extract spatial coordinates from DNF activity
# input : DNF potentials as a 2D numpy array
# output : 2D spatial coordinates as a tuple
def space_to_rate_code(neuronal_population):
    normalization = 0.0
    position = np.array([0.0, 0.0])

    if neuronal_population.ndim == 2:
        for index, value in np.ndenumerate(neuronal_population):
            normalization += value

        if normalization != 0.0:
            for index, value in np.ndenumerate(neuronal_population):
                position += np.array(index) * value
            position = position / normalization
            return tuple(position)
        else:
            # print ("Warning : normalization constant = 0")
            return tuple((None,) * neuronal_population.ndim)


def normalized_space_to_rate_code(neuronal_population, shape):
    pos = np.array(space_to_rate_code(neuronal_population))
    shape = np.array(shape)
    return pos / shape


def absolute_error(true_val, estimated_val):
    if true_val is None or estimated_val is None:
        return None
    else:
        return np.abs(true_val - estimated_val)


def random_noise(shape=(width, height), max_noise_amplitude=0.01):
    return max_noise_amplitude * (np.random.random(size=shape))
