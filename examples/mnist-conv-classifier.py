#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import theanets
import theano

from utils import load_mnist, plot_layers

#theano.config.exception_verbosity = 'high'
#theano.config.compute_test_value = 'warn'
train, valid, _ = load_mnist(labels=True)

print np.shape(np.asarray(train[0])), np.shape(np.asarray(valid[0]))

N = 16

rng = np.random.RandomState(23455)

e = theanets.Experiment(
    theanets.Classifier,
    cnn = 1,
    input2d = 1,
    rng = rng, 
    input_dim=(28,28),
    feature_maps=(1,20,50),
    filter_size=(5,5,5,5),
    max_pool=(2,2,2,2), 
    layers=(500, 10),
    train_batches=100,
)
e.run(train, valid)

plot_layers(e.network.weights)
plt.tight_layout()
plt.show()
