from chainer import datasets
import numpy as np
import playground

train, test = datasets.get_mnist()

counts = np.zeros(10, dtype=np.int32)
sums = np.zeros((10, 28 * 28), dtype=np.float32)
for x, y in train:
    counts[y] += 1
    sums[y] += x

for i in range(10):
    playground.print_mnist(sums[i] / counts[i])

