from chainer import functions as F
from chainer import links as L
from chainer import Variable
import numpy as np


lin = L.Linear(5, 2)
x = Variable(np.ones((3, 5), dtype=np.float32))
y1 = lin(x)

y2 = F.sigmoid(lin(x))
print(y2.data)

