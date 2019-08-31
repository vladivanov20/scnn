import numpy as np

true_w = np.array([1, 2, 3, 4, 5])
d = 5
points = []

for i in range(10000):
    x = np.random.randn(5)
    y = true_w.dot(x) + np.random.randn()
    points.append((x, y))

def sF(w, i):  # Loss(x_i, y_i, w)
    x, y = points[i]
    return (w.dot(x) - y)**2


def sdF(w, i):  # gradient
    x, y = points[i]
    return 2 * (w.dot(x) - y) * x


def stochasticGradientDescent(sF, sdF, d, n):
    w = np.zeros(d)
    num_updates = 0
    eta = 0.01
    for t in range(500):
        for i in range(n):
            value = sF(w, i)
            gradient = sdF(w, i)
            num_updates += 1
            eta = 1.0 / num_updates
            w = w - eta * gradient  # Key: take a step
        print('numIters = %d, weight = %s, Training Loss Value = %s, gradient = %s' % (t, w, value, gradient))


stochasticGradientDescent(sF, sdF, d, len(points))
