import numpy as np


def moving_average(a, n=3):
	return np.convolve(a, np.ones((n,)) / n, mode='valid')


def moving_average2(a, n=3):
	ret = np.cumsum(a, dtype=float)
	ret[n:] = ret[n:] - ret[:-n]
	return ret[n - 1:] / n


myX = np.arange(0, 10, 1)  # query points
Y =

X = moving_average(myX, n=5)
X2 = moving_average2(myX, n=5)


print(myX, X, X2)
print(len(myX), len(X))
