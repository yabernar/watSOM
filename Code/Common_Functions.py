import numpy as np

img_dims = [5, 5]

# kernel_size = np.asarray(img_dims)*2-1
# print(kernel_size)
# kernel = np.zeros(kernel_size, dtype=float)
# for i in range(len(kernel[0])):
#     for j in range(len(kernel[1])):
#         d = np.sqrt(((i - kernel_size[0]//2)/ kernel_size[0]) ** 2 + ((j - kernel_size[0]//2)/ kernel_size[0]) ** 2) / np.sqrt(0.5)
#         kernel[i, j] = np.exp(-d**2/2)
# print(kernel)
# kernel.flatten()

kernel_size = np.array([np.prod(img_dims), np.prod(img_dims)])
# print(kernel_size)
kernel = np.zeros(kernel_size, dtype=float)
for i in range(kernel_size[0]):
    for j in range(kernel_size[1]):
        a = np.array([(i%img_dims[0])/img_dims[0], (i//img_dims[0])/img_dims[1]])
        b = np.array([(j%img_dims[0])/img_dims[0], (j//img_dims[0])/img_dims[1]])
        d = np.sqrt(np.sum((a-b)**2))
        kernel[i, j] = np.exp(-d**2/2)
print(kernel)
# kernel.flatten()


def quadratic_distance(x, y):
    assert np.array_equal(np.array(x.shape), np.array(y.shape))
    return np.sum((x - y) ** 2)


def manhattan_distance(x, y):
    return np.sum(np.abs(x - y))


def oddr_to_cube(hex):
    x = hex[0] - (hex[1] - (hex[1]&1)) // 2
    z = hex[1]
    y = -x-z
    return x, y, z


def cube_distance(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2]))


def hexagonal_distance(x, y):
    a = oddr_to_cube(x)
    b = oddr_to_cube(y)
    return cube_distance(a, b)

# As defined in "On the Euclidean Distance of Images", Liwei Wang, Yan Zhang, and Jufu Feng
# Naive implementation
def image_euclidean_distance(x, y, img_dims=img_dims):
    sum = 0
    for i in range(len(x)):
        pixel = 0
        for j in range(len(y)):
            distance = np.sqrt(((i%img_dims[0])/img_dims[0] - (j%img_dims[0])/img_dims[0])**2 + ((i//img_dims[1])/img_dims[1] - (j//img_dims[1])/img_dims[
                1])**2)/np.sqrt(0.5)
            pixel += np.exp(-distance ** 2 / 2) * (x[i] - y[i]) * (x[j] - y[j])
        sum += pixel
    return np.sqrt(sum/(2*np.pi))


def fast_ied(x, y, img_dims=img_dims):
    a = x-y
    sum = np.sum(np.outer(a, a) * kernel)
    return sum


def gaussian(d, sigma):
    return np.exp(-((d / sigma) ** 2) / 2) / sigma
    # Between 0 and 1/sig


def normalized_gaussian(d, sigma):
    return np.exp(-((d / sigma) ** 2) / 2)
    # Between 0 and 1


def euclidean_norm(x, y):
    assert np.array_equal(np.array(x.shape), np.array(y.shape))
    return np.sqrt(np.sum((x - y) ** 2))


def normalized_euclidean_norm(x, y):
    assert np.array_equal(np.array(x.shape), np.array(y.shape))
    return np.sqrt(np.sum((x - y) ** 2))/np.sqrt(x.shape[0])


if __name__ == '__main__':
    img_dims = [5, 5]
    a = np.zeros(img_dims).flatten()
    b = np.zeros(img_dims).flatten()
    c = np.zeros(img_dims).flatten()
    d = np.zeros(img_dims).flatten()
    a[3] = 1
    for i in range(len(b)):
        b[i] = 1
        c[i] = quadratic_distance(a, b)
        d[i] = fast_ied(a, b, img_dims)
        b[i] = 0
    print(c.reshape(img_dims))
    print(d.reshape(img_dims))
