import chainer
from chainer import functions as F
from chainer import links as L
import numpy as np


class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)    # n_units -> n_out

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


batchsize = 100
train, test = chainer.datasets.get_mnist()
train_iter = chainer.iterators.SerialIterator(train, batchsize)
test_iter = chainer.iterators.SerialIterator(
    test, batchsize, repeat=True, shuffle=False)

model = MLP(784, 10)
opt = chainer.optimizers.Adam()
opt.setup(model)

train_num = len(train)
test_num = len(test)

epoch_num = 10
for epoch in range(epoch_num):
    train_loss_sum = 0
    train_accuracy_sum = 0
    for i in range(0, train_num, batchsize):
        batch = train_iter.next()
        x = np.asarray([s[0] for s in batch])
        t = np.asarray([s[1] for s in batch])
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        model.cleargrads()
        loss.backward()
        opt.update()
        train_loss_sum += loss.data
        train_accuracy_sum += F.accuracy(y, t).data
    print(train_loss_sum, train_accuracy_sum)

    test_loss_sum = 0
    test_accuracy_sum = 0
    for i in range(0, test_num, batchsize):
        batch = test_iter.next()
        x = np.asarray([s[0] for s in batch])
        t = np.asarray([s[1] for s in batch])
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        test_loss_sum += loss.data
        test_accuracy_sum += F.accuracy(y, t).data

    print("%5d %.5f %.5f %.5f %.5f" %
          (epoch,
           train_loss_sum / train_num,
           train_accuracy_sum / train_num * 100,
           test_loss_sum / test_num,
           test_accuracy_sum / test_num * 100))

