import chainer
from chainer import datasets
from chainer import functions as F
from chainer import links as L
from chainer import training
from chainer.training import extensions


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


batchsize = 10
train, test = datasets.get_mnist()
train = datasets.SubDataset(train, 0, 100)
test = datasets.SubDataset(test, 0, 100)
train_iter = chainer.iterators.SerialIterator(train, batchsize)
test_iter = chainer.iterators.SerialIterator(test, batchsize,
                                             repeat=False, shuffle=False)

model = L.Classifier(MLP(784, 10))
opt = chainer.optimizers.Adam()
opt.setup(model)

epoch = 2

# Set up a trainer
updater = training.StandardUpdater(train_iter, opt, device=-1)
trainer = training.Trainer(updater, (epoch, 'epoch'), out='/tmp/result')

# Evaluate the model with the test dataset for each epoch
trainer.extend(extensions.Evaluator(test_iter, model, device=-1))

# Dump a computational graph from 'loss' variable at the first iteration
# The "main" refers to the target link of the "main" optimizer.
# trainer.extend(extensions.dump_graph('main/loss'))

# Take a snapshot at each epoch
# trainer.extend(extensions.snapshot(), trigger=(epoch, 'epoch'))

# Write a log of evaluation statistics for each epoch
trainer.extend(extensions.LogReport())

# Print selected entries of the log to stdout
# Here "main" refers to the target link of the "main" optimizer again, and
# "validation" refers to the default name of the Evaluator extension.
# Entries other than 'epoch' are reported by the Classifier link, called by
# either the updater or the evaluator.
trainer.extend(extensions.PrintReport(
    ['epoch', 'main/loss', 'validation/main/loss',
     'main/accuracy', 'validation/main/accuracy']))

# Print a progress bar to stdout
# trainer.extend(extensions.ProgressBar())

resume = False
if resume:
    # Resume from a snapshot
    chainer.serializers.load_npz(resume, trainer)

# Run the training
trainer.run()

