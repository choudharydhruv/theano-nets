#!/usr/bin/env python

import matplotlib.pyplot as plt
import theanets

from utils import load_mnist, plot_layers


train, valid, _ = load_mnist(labels=True)

N = 16

e = theanets.Experiment(
    theanets.Classifier,
    cnn = 1,
    input2d = 1,
    input_dim=(28,28),
    feature_maps=(1,20,50),
    filter_size=(5,5,5,5), 
    layers=(500, 10),
    train_batches=100,
)
e.run(train, valid)

plot_layers(e.network.weights)
plt.tight_layout()
plt.show()
