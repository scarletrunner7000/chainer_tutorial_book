import chainer
from chainer import datasets
from chainer import functions as F
from chainer import links as L
from chainer import optimizers
from chainer import training
from chainer.training import extensions
import numpy as np

import playground


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


# create model
model = L.Classifier(MLP(100, 10))

# load dataset
train_full, test_full = datasets.get_mnist()
train = datasets.SubDataset(train_full, 0, 1000)
test = datasets.SubDataset(test_full, 0, 1000)

# Set up a iterator
batchsize = 100
train_iter = chainer.iterators.SerialIterator(train, batchsize)
test_iter = chainer.iterators.SerialIterator(test, batchsize,
                                             repeat=False, shuffle=False)

# Set up an optimizer
opt = chainer.optimizers.Adam()
opt.setup(model)

# Set up an updater
updater = training.StandardUpdater(train_iter, opt, device=-1)

# Set up a trainer
epoch = 10
trainer = training.Trainer(updater, (epoch, 'epoch'), out='/tmp/result')
trainer.extend(extensions.Evaluator(test_iter, model, device=-1))
trainer.extend(extensions.LogReport(trigger=(1, "epoch")))
trainer.extend(extensions.PrintReport(
        ['epoch',
         'main/accuracy', 'validation/main/accuracy']), trigger=(1, "epoch"))

# Run the trainer
trainer.run()

# Check the result
x, y = test[np.random.randint(len(test))]
playground.print_mnist(x)
pred = F.softmax(model.predictor(x.reshape(1, 784))).data
print("Prediction: ", np.argmax(pred))
print("Correct answer: ", y)

