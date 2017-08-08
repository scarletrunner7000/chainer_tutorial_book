from chainer import Chain
from chainer import ChainList
from chainer import links as L


class MyChain(Chain):
    def __init__(self):
        super(MyChain, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(4, 3)
            self.l2 = L.Linear(3, 2)

    def __call__(self, x):
        h = self.l1(x)
        return self.l2(h)


class MyChainList(ChainList):
    def __init__(self):
        super(MyChainList, self).__init__(
            L.Linear(4, 3),
            L.Linear(3, 2),
        )

    def __call__(self, x):
        h = self[0](x)
        return self[1](h)


class MyChainListN(ChainList):
    def __init__(self, n):
        links = [L.Linear(3, 3) for i in range(n)]
        self._n = n
        super(MyChainListN, self).__init__(*links)

    def __call__(self, x):
        h = x
        for i in range(self._n):
            h = self[0](h)
        return h

c = MyChain()
c2 = MyChainList()
c3 = MyChainListN(100)

