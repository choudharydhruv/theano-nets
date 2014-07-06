# Copyright (c) 2012-2014 Leif Johnson <leif@leifjohnson.net>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

'''This module contains a number of classes for modeling neural nets in Theano.
'''

import climate
import pickle
import functools
import gzip
import numpy as np
import theano
import theano.tensor as TT
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
import copy

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from operator import add, sub, floordiv, mul
logging = climate.get_logger(__name__)

FLOAT = theano.config.floatX

import warnings
warnings.filterwarnings("ignore")

class Network(object):
    '''The network class encapsulates a fully-connected feedforward net.

    In addition to defining standard functionality for feedforward nets, there
    are also many options for specifying topology and regularization, several of
    which must be provided to the constructor at initialization time.

    Parameters
    ----------
    layers : sequence of int
        A sequence of integers specifying the number of units at each layer. As
        an example, layers=(10, 20, 3) has one "input" layer with 10 units, one
        "hidden" layer with 20 units, and one "output" layer with 3 units. That
        is, inputs should be of length 10, and outputs will be of length 3.

    activation : string
        The name of an activation function to use on hidden network units.

    rng : theano RandomStreams object, optional
        Use a specific Theano random number generator. A new one will be created
        if this is None.

    input_noise : float, optional
        Standard deviation of desired noise to inject into input.

    hidden_noise : float, optional
        Standard deviation of desired noise to inject into hidden unit
        activation output.

    input_dropouts : float in [0, 1], optional
        Proportion of input units to randomly set to 0.

    hidden_dropouts : float in [0, 1], optional
        Proportion of hidden unit activations to randomly set to 0.

    decode : positive int, optional
        Any of the hidden layers can be tapped at the output. Just specify a
        value greater than 1 to tap the last N hidden layers. The default is 1,
        which decodes from just the last layer.

    tied_weights : bool, optional
        Construct decoding weights using the transpose of the encoding weights
        on corresponding layers. If not True, decoding weights will be
        constructed using a separate weight matrix.

    Attributes
    ----------
    weights : list of Theano shared variables
        Theano shared variables containing network connection weights.

    biases : list of Theano shared variables
        Theano shared variables containing biases for hidden and output units.

    hiddens : list of Theano variables
        Computed Theano variables for the state of hidden units in the network.
    '''

    def __init__(self, layers, activation, **kwargs):
        self.layers = tuple(layers)
        self.hiddens = []
        self.weights = []
        self.biases = []
        self.filter_shapes = []
        self.pool_sizes = []

        self.rng = kwargs.get('rng') or RandomStreams()
        self.tied_weights = bool(kwargs.get('tied_weights'))
        self.cnn = bool(kwargs.get('cnn'))
        self.input2d = bool(kwargs.get('input2d'))
        self.featuremaps = np.asarray(kwargs.get('feature_maps'))
        self.input_dim = np.asarray(kwargs.get('input_dim'))
        self.filter_shape = np.asarray(kwargs.get('filter_size'))
        self.pool_size = np.asarray(kwargs.get('max_pool'))
	self.batch_size = int(kwargs.get('batch_size', 64))

        # x is a proxy for our network's input, and y for its output.
        #self.x = TT.dtensor4('x')
        self.x = TT.matrix('x')
        #self.x.tag.test_value = np.random.rand(64, 53550)
        

        activation = self._build_activation(activation)
        if hasattr(activation, '__theanets_name__'):
            logging.info('hidden activation: %s', activation.__theanets_name__)

        # ensure that --layers is compatible with --tied-weights.
        sizes = layers[:-1]
        if self.tied_weights:
            error = 'with --tied-weights, --layers must be an odd-length palindrome'
            assert len(layers) % 2 == 1, error
            k = len(layers) // 2
            encode = np.asarray(layers[:k])
            decode = np.asarray(layers[k+1:])
            assert np.allclose(encode - decode[::-1], 0), error
            sizes = layers[:k+1]

        if(self.cnn):
            _, parameter_count = self._create_convolution_forward_map(activation, **kwargs)
        else:
            _, parameter_count = self._create_forward_map(sizes, activation, **kwargs)

        # set up the "decoding" computations from layer activations to output.
        w = len(self.weights)
        #print self.weights
        if self.tied_weights:
            for i in range(w - 1, -1, -1):
                h = self.hiddens[-1]
                a, b = self.weights[i].get_value(borrow=True).shape
                logging.info('tied weights from layer %d: %s x %s', i, b, a)
                # --tied-weights implies --no-learn-biases (biases are zero).
                self.hiddens.append(TT.dot(h, self.weights[i].T))
        else:
            n = layers[-1]
            decoders = []
            for i in range(w - 1, w - 1 - kwargs.get('decode', 1), -1):
                b = self.biases[i].get_value(borrow=True).shape[0]
                Di, _, count = self._create_layer(b, n, 'out_%d' % i)
                parameter_count += count - n
                decoders.append(TT.dot(self.hiddens[i], Di))
                self.weights.append(Di)
            parameter_count += n
            bias = theano.shared(np.zeros((n, ), FLOAT), name='bias_out')
            self.biases.append(bias)
            self.hiddens.append(sum(decoders) + bias)

        logging.info('%d total network parameters', parameter_count)

        self.y = self.hiddens.pop()
        self.updates = {}

    @property
    def inputs(self):
        '''Return a list of Theano input variables for this network.'''
        return [self.x]

    @property
    def monitors(self):
        '''Generate a sequence of name-value pairs for monitoring the network.
        '''
        yield 'error', self.cost
        for i, h in enumerate(self.hiddens):
            yield 'h{}<0.1'.format(i+1), (abs(h) < 0.1).mean()
            yield 'h{}<0.9'.format(i+1), (abs(h) < 0.9).mean()

    @staticmethod
    def _create_layer(a, b, suffix, sparse=None):
        '''Create a layer of weights and bias values.

        Parameters
        ----------
        a : int
            Number of rows of the weight matrix -- equivalently, the number of
            "input" units that the weight matrix connects.
        b : int
            Number of columns of the weight matrix -- equivalently, the number
            of "output" units that the weight matrix connects.
        suffix : str
            A string suffix to use in the Theano name for the created variables.
            This string will be appended to "W_" (for the weights) and "b_" (for
            the biases) parameters that are created and returned.
        sparse : float in (0, 1)
            If given, ensure that the weight matrix for the layer has only this
            proportion of nonzero entries.

        Returns
        -------
        weight : Theano shared array
            A shared array containing Theano values representing the weights
            connecting each "input" unit to each "output" unit.
        bias : Theano shared array
            A shared array containing Theano values representing the bias
            values on each of the "output" units.
        count : int
            The number of parameters that are included in the returned
            variables.
        '''
        arr = np.random.randn(a, b) / np.sqrt(a + b)
        if sparse is not None:
            arr *= np.random.binomial(n=1, p=sparse, size=(a, b))
        weight = theano.shared(arr.astype(FLOAT), name='W_{}'.format(suffix))
        bias = theano.shared(np.zeros((b, ), FLOAT), name='b_{}'.format(suffix))
        logging.info('weights for layer %s: %s x %s', suffix, a, b)
        return weight, bias, (a + 1) * b

    def __find_shapes(self, img_shape, num_layers):
        self.layer_shapes = []
        next_shape = list(img_shape)
        self.layer_shapes.append(list(img_shape))
   
        print "Number of Convolution layers: ", num_layers 
        for i in range(0, num_layers):
            filter_shape = list(self.filter_shapes[i])
            print "Layer ", i
            print "    Filter shape: ", filter_shape
            print "    Input Shape: ", next_shape
            print "    Pooling dimensions: ", self.pool_sizes[i]

            # Convolution.
            next_shape[2:] = map(sub, next_shape[2:], filter_shape[2:])
            next_shape[2:] = map(add, next_shape[2:], [1, 1])
            next_shape[1] = filter_shape[0]

            # Max pooling.
            next_shape[2:] = map(floordiv, next_shape[2:], self.pool_sizes[i])
            print "    Output Shape: ", next_shape

            # The fun copying stuff is so new changes to next_shape don't modify
            # references already in the list.
            self.layer_shapes.append([])
            for num in next_shape:
                cop = copy.deepcopy(num)
                self.layer_shapes[-1].append(cop)

    def _create_forward_map(self, sizes, activation, **kwargs):
        '''Set up a computation graph to map the input to layer activations.

        Parameters
        ----------
        sizes : list of int
            A list of the number of nodes in each feedforward hidden layer.

        activation : callable
            The activation function to use on each feedforward hidden layer.

        input_noise : float, optional
            Standard deviation of desired noise to inject into input.

        hidden_noise : float, optional
            Standard deviation of desired noise to inject into hidden unit
            activation output.

        input_dropouts : float in [0, 1], optional
            Proportion of input units to randomly set to 0.

        hidden_dropouts : float in [0, 1], optional
            Proportion of hidden unit activations to randomly set to 0.

        Returns
        -------
        parameter_count : int
            The number of parameters created in the forward map.
        '''
        parameter_count = 0
        z = self._add_noise(
            self.x,
            kwargs.get('input_noise', 0.),
            kwargs.get('input_dropouts', 0.))


        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            Wi, bi, count = self._create_layer(a, b, i)
            parameter_count += count
            self.hiddens.append(self._add_noise(
                activation(TT.dot(z, Wi) + bi),
                kwargs.get('hidden_noise', 0.),
                kwargs.get('hidden_dropouts', 0.)))
            self.weights.append(Wi)
            self.biases.append(bi)
            z = self.hiddens[-1]
        return z, parameter_count

    def _create_convolution_forward_map(self, activation, **kwargs):
        '''Set up a computation graph to map the input to layer activations.

        Parameters
        ----------
        layers : list of int
            A list of the number of nodes in each feedforward hidden layer after the convolution layers.

        activation : callable
            The activation function to use on each feedforward hidden layer.

        Returns
        -------
        parameter_count : int
            The number of parameters created in the forward map.
        '''

        batch_size = self.batch_size
        ps = kwargs.get('maxpool', 2)

        parameter_count = 0
        z = self._add_noise(
            self.x,
            kwargs.get('input_noise', 0.),
            kwargs.get('input_dropouts', 0.))
   
        sizes = self.layers[:-1]

        '''
        If input is 2D e.g. images convlayers is the sequence of feature map sizes.
        Each size is a 4D tensor batch_size, input_feature_map, height, width
        '''
        #Dimensions of input image
        height = self.input_dim[0]
        width = self.input_dim[1]

        #Array of feature maps per layer
        fmaps = self.featuremaps

        #Number of input feature maps
        input_fmap = fmaps[0]

        #Populating self.filter_shapes       
        flt_arr = self.filter_shape.reshape(len(self.filter_shape)/2, 2)
        pool_arr = self.pool_size.reshape(len(self.pool_size)/2, 2)
        for i, (i1, i2, (i3, i4)) in enumerate(zip(fmaps[1:], fmaps[:-1], flt_arr)):
            self.filter_shapes.append((i1,i2,i3,i4))
        for i, (i1, i2) in enumerate(pool_arr):
            self.pool_sizes.append((i1, i2))

        img_shape = (batch_size, input_fmap, height, width)
        self.__find_shapes(img_shape, len(fmaps[1:]))

        print "Shapes of all convolution layers", self.layer_shapes
        #print "Type of x", type(self.x)   

        for i, (i1, i2, (i3, i4)) in enumerate(zip(fmaps[1:], fmaps[:-1], flt_arr)):
            self._add_conv_layer((i1, i2, i3, i4), self.layer_shapes[i])

        layers = list(self.layers[:-1])
        flat_size = reduce(mul, self.layer_shapes[-1], 1)
        flat_size = np.prod(self.layer_shapes[-1][1:])
        #print flat_size
        layers.insert(0, flat_size)
        self._make_graph(img_shape, activation)
        parameter_count = 0
        z = self._add_noise(
            self.conv_output,
            kwargs.get('input_noise', 0.),
            kwargs.get('input_dropouts', 0.))

        for i, (a, b) in enumerate(zip(layers[:-1], layers[1:])):
            Wi, bi, count = self._create_layer(a, b, i)
            parameter_count += count
            self.hiddens.append(self._add_noise(
                activation(TT.dot(z, Wi) + bi),
                kwargs.get('hidden_noise', 0.),
                kwargs.get('hidden_dropouts', 0.)))
            self.weights.append(Wi)
            self.biases.append(bi)
            z = self.hiddens[-1]
        return z, parameter_count
        '''
        layer_input = z.reshape((batch_size, input_fmap, height, width))
        j=0

        # Construct the convolutional pooling layers:
        # filtering reduces the image size to (xdim-f1+1,ydim-f2+1)
        # maxpooling reduces this further to (dim1/ps,dim2/ps) = (12,12)
        # 4D output tensor is thus of shape (20,20,12,12)
        for i, (a, (f1, f2)) in enumerate(zip(fmaps[1:], flt_arr)):
            clayer = ConvPoolLayer(self.rng, activation, i, inp=layer_input, image_shape=(batch_size, input_fmap, height, width), filter_shape=(a, input_fmap, f1, f2), pool=(ps, ps))
            #print "Conv IShapes", layer_input.shape.eval()
            layer_input = clayer.output
            input_fmap = a
            height = (height - f1 + 1)/ps
            width = (width - f2 + 1)/ps
            self.hiddens.append(self._add_noise(clayer.output, kwargs.get('hidden_noise', 0.), kwargs.get('hidden_dropouts', 0.)))
            self.weights.append(clayer.W)
            self.biases.append(clayer.b)
            parameter_count += clayer.param_count
	    j=i+1
        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size,num_inputs) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (batchsize, num_inputs) 
        hlayer_input = layer_input.flatten(2) 

        #The input dimension to the hidden nueron layer is num_inputs or num pixels 
        #which is the 2nd dimension of hlayer_input
        sizes = self.layers[:-1] 

        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            Wi, bi, count = self._create_layer(a, b, i+j)
            parameter_count += count
            self.hiddens.append(self._add_noise(
                activation(TT.dot(hlayer_input, Wi) + bi),
                kwargs.get('hidden_noise', 0.),
                kwargs.get('hidden_dropouts', 0.)))
            self.weights.append(Wi)
            self.biases.append(bi)
            hlayer_input = self.hiddens[-1]
        return hlayer_input, parameter_count
        '''


    def _add_conv_layer(self, filter_shape, img_shape):
    
        #print filter_shape
        #print img_shape
        if filter_shape[1] != img_shape[1]:
            raise RuntimeError("Input feature maps must be the same.")

        rng = np.random.RandomState()
        fan_in = np.prod(filter_shape[1:])
        weight_values = np.asarray(self.rng.uniform(
            low = -np.sqrt(3. / fan_in),
            high = np.sqrt(3. / fan_in),
            size = filter_shape), dtype = theano.config.floatX)
        weights = theano.shared(value = weight_values, name = "weights")
        self.weights.append(weights)

        bias_values = np.zeros((filter_shape[0],), dtype = theano.config.floatX)
        biases = theano.shared(value = bias_values, name = "biases")
        self.biases.append(biases)

    def _make_graph(self, img_shape, activation):
        #self._inputs = TT.dtensor4("inputs")
        layer_outputs = self.x.reshape(img_shape)
        for i in range(0, len(self.weights)):
            # Perform the convolution.
            conv_out = conv.conv2d(layer_outputs, self.weights[i],
              filter_shape = self.filter_shapes[i],
              image_shape = self.layer_shapes[i])
      
            # Downsample the feature maps.
            pooled_out = downsample.max_pool_2d(conv_out, self.pool_sizes[i],
              ignore_border = True)

            # Account for the bias. Since it is a vector, we first need to reshape it
            # to (1, n_filters, 1, 1).
            layer_outputs = activation(pooled_out + self.biases[i].dimshuffle("x", 0, "x", "x"))
            self.hiddens.append(layer_outputs)

        # Concatenate output maps into one big matrix where each row is the
        # concatenation of all the feature maps from one item in the batch.
        next_shape = self.layer_shapes[i + 1]
        new_shape = (next_shape[0], reduce(mul, next_shape[1:], 1))
        print "Hidden unit shape for the fully connected part: ", new_shape
        self.conv_output = layer_outputs.reshape(new_shape)

    def _add_noise(self, x, sigma, rho):
        '''Add noise and dropouts to elements of x as needed.

        Parameters
        ----------
        x : Theano array
            Input array to add noise and dropouts to.
        sigma : float
            Standard deviation of gaussian noise to add to x. If this is 0, then
            no gaussian noise is added to the values of x.
        rho : float, in [0, 1]
            Fraction of elements of x to set randomly to 0. If this is 0, then
            no elements of x are set randomly to 0. (This is also called
            "salt-and-pepper noise" or "dropouts" in the research community.)

        Returns
        -------
        Theano array
            The parameter x, plus additional noise as specified.
        '''
        if sigma > 0 and rho > 0:
            noise = self.rng.normal(size=x.shape, std=sigma)
            mask = self.rng.binomial(size=x.shape, n=1, p=1-rho, dtype=FLOAT)
            return mask * (x + noise)
        if sigma > 0:
            return x + self.rng.normal(size=x.shape, std=sigma)
        if rho > 0:
            mask = self.rng.binomial(size=x.shape, n=1, p=1-rho, dtype=FLOAT)
            return mask * x
        return x

    def _compile(self):
        '''If needed, compile the Theano function for this network.'''
        if getattr(self, '_compute', None) is None:
            self._compute = theano.function(
                [self.x], self.hiddens + [self.y], updates=self.updates)

    def _build_activation(self, act=None):
        '''Given an activation description, return a callable that implements it.

        Parameters
        ----------
        activation : string
            A string description of an activation function to use.

        Returns
        -------
        callable(float) -> float :
            A callable activation function.
        '''
        def compose(a, b):
            c = lambda z: b(a(z))
            c.__theanets_name__ = '%s(%s)' % (b.__theanets_name__, a.__theanets_name__)
            return c
        if '+' in act:
            return functools.reduce(
                compose, (self._build_activation(a) for a in act.split('+')))
        options = {
            'tanh': TT.tanh,
            'linear': lambda z: z,
            'logistic': TT.nnet.sigmoid,
            'sigmoid': TT.nnet.sigmoid,
            'softplus': TT.nnet.softplus,

            # shorthands
            'relu': lambda z: TT.maximum(0, z),
            'trec': lambda z: z * (z > 1),
            'tlin': lambda z: z * (abs(z) > 1),

            # modifiers
            'rect:max': lambda z: TT.minimum(1, z),
            'rect:min': lambda z: TT.maximum(0, z),

            # normalization
            'norm:dc': lambda z: (z.T - z.mean(axis=1)).T,
            'norm:max': lambda z: (z.T / TT.maximum(1e-10, abs(z).max(axis=1))).T,
            'norm:std': lambda z: (z.T / TT.maximum(1e-10, TT.std(z, axis=1))).T,
            }
        for k, v in options.items():
            v.__theanets_name__ = k
        try:
            return options[act]
        except KeyError:
            raise KeyError('unknown activation %r' % act)

    def params(self, **kwargs):
        '''Return a list of the Theano parameters for this network.'''
        params = []
        params.extend(self.weights)
        if getattr(self, 'tied_weights', False) or kwargs.get('no_learn_biases'):
            # --tied-weights implies --no-learn-biases.
            pass
        else:
            params.extend(self.biases)
        return params

    def feed_forward(self, x):
        '''Compute a forward pass of all activations from the given input.

        Parameters
        ----------
        x : ndarray
            An array containing data to be fed into the network.

        Returns
        -------
        list of ndarray
            Returns the activation values of each layer in the the network when
            given input `x`.
        '''
        #print "Shape of batch", x.shape()

        self._compile()
        return self._compute(x)

    def predict(self, x):
        '''Compute a forward pass of the inputs, returning the net output.

        Parameters
        ----------
        x : ndarray
            An array containing data to be fed into the network.

        Returns
        -------
        ndarray
            Returns the values of the network output units when given input `x`.
        '''
        y_pred = []
	x_new = [x]
        size = self.batch_size
        overshoot = size - (len(x[:,0])%size)
	#print "Overshoot is ", overshoot 
        if overshoot != 0:
	    x_new.append(x[0:overshoot])
	x_new = np.vstack(x_new)
	#print x_new.shape
	assert len(x_new[:,0])%size is 0
 
        for i in range(0, len(x_new[:,0]), size):
	    pred = self.feed_forward(x_new[i:i + size])[-1]
	    #print pred[:,0]
	    y_pred.append(pred[:,0])
	#print y_pred
        y_pred = np.hstack(y_pred)
	overshoot = (-1)*overshoot
	#print "Padded shape", y_pred.shape 
	y_pred = y_pred[:overshoot]
	print "Final returned shape of prediction", y_pred.shape 
        return y_pred
        #return self.feed_forward(x)[-1]
   
    __call__ = predict

    def save(self, filename):
        '''Save the parameters of this network to disk.

        Parameters
        ----------
        filename : str
            Save the parameters of this network to a pickle file at the named
            path. If this name ends in ".gz" then the output will automatically
            be gzipped; otherwise the output will be a "raw" pickle.
        '''
        opener = gzip.open if filename.lower().endswith('.gz') else open
        handle = opener(filename, 'wb')
        pickle.dump(
            dict(weights=[p.get_value().copy() for p in self.weights],
                 biases=[p.get_value().copy() for p in self.biases],
                 ), handle, -1)
        handle.close()
        logging.info('%s: saved model parameters', filename)

    def load(self, filename):
        '''Load the parameters for this network from disk.

        Parameters
        ----------
        filename : str
            Load the parameters of this network from a pickle file at the named
            path. If this name ends in ".gz" then the input will automatically
            be gunzipped; otherwise the input will be treated as a "raw" pickle.
        '''
        opener = gzip.open if filename.lower().endswith('.gz') else open
        handle = opener(filename, 'rb')
        params = pickle.load(handle)
        for target, source in zip(self.weights, params['weights']):
            target.set_value(source)
        for target, source in zip(self.biases, params['biases']):
            target.set_value(source)
        handle.close()
        logging.info('%s: loaded model parameters', filename)

    def J(self, weight_l1=0, weight_l2=0, hidden_l1=0, hidden_l2=0, contractive_l2=0, **unused):
        '''Return a variable representing the cost or loss for this network.

        Parameters
        ----------
        weight_l1 : float, optional
            Regularize the L1 norm of unit connection weights by this constant.
        weight_l2 : float, optional
            Regularize the L2 norm of unit connection weights by this constant.
        hidden_l1 : float, optional
            Regularize the L1 norm of hidden unit activations by this constant.
        hidden_l2 : float, optional
            Regularize the L2 norm of hidden unit activations by this constant.
        contractive_l2 : float, optional
            Regularize model using the Frobenius norm of the hidden Jacobian.

        Returns
        -------
        Theano variable
            A variable representing the overall cost value of this network.
        '''
        cost = self.cost
        if weight_l1 > 0:
            cost += weight_l1 * sum(abs(w).sum() for w in self.weights)
        if weight_l2 > 0:
            cost += weight_l2 * sum((w * w).sum() for w in self.weights)
        if hidden_l1 > 0:
            cost += hidden_l1 * sum(abs(h).mean(axis=0).sum() for h in self.hiddens)
        if hidden_l2 > 0:
            cost += hidden_l2 * sum((h * h).mean(axis=0).sum() for h in self.hiddens)
        if contractive_l2 > 0:
            cost += contractive_l2 * sum(
                TT.sqr(TT.grad(h.mean(axis=0).sum(), self.x)).sum() for h in self.hiddens)
        return cost

class ConvPoolLayer(object):
    def __init__(self, rng, activation, suffix, inp, filter_shape, image_shape, pool=(2,2)):
        """
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4 or theano.tensor.dtensor5
        :param input: symbolic image tensor, of shape img_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters(output feature maps), num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type pool: tuple or list of length 2
        :param pool: the downsampling (pooling) factor (#rows,#cols)

        :type activation: callable activation function
        :param activation: type of activation TanH|Sigmoid...
        """
        assert image_shape[1] == filter_shape[1]
        self.input = inp

        # initialize weight values: the fan-in of each hidden neuron is
        # restricted by the size of the receptive fields.
        fan_in =  np.prod(filter_shape[1:])
        print filter_shape, fan_in
        W_values = np.asarray(rng.uniform( low=-np.sqrt(3./fan_in), high=np.sqrt(3./fan_in), size=filter_shape), dtype=FLOAT)
        self.W = theano.shared(value=W_values, name='W_{}'.format(suffix))

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = np.zeros((filter_shape[0],), dtype=FLOAT)
        self.b = theano.shared(value=b_values, name='b_{}'.format(suffix))
        #print W_values, b_values

        # convolve input feature maps with filters
        conv_out = conv.conv2d(inp, self.W,
                filter_shape=filter_shape, image_shape=image_shape)

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(conv_out, pool, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will thus
        # be broadcasted across mini-batches and feature map width & height
        self.output = activation(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]
        self.param_count = (filter_shape[0]+1)*filter_shape[2]*filter_shape[3]
        print "Weights Shape ", W_values.shape, b_values.shape 

        logging.info('featuremaps for conv layer %s: %s x %s', suffix, image_shape[1], filter_shape[0])

class Autoencoder(Network):
    '''An autoencoder attempts to reproduce its input.'''

    @property
    def cost(self):
        err = self.y - self.x
        return TT.mean((err * err).sum(axis=1))

    def encode(self, x, layer=None, sample=False):
        '''Encode a dataset using the hidden layer activations of our network.

        Parameters
        ----------
        x : ndarray
            A dataset to encode. Rows of this dataset capture individual data
            points, while columns represent the variables in each data point.

        layer : int, optional
            The index of the hidden layer activation to use. By default, we use
            the "middle" hidden layer---for example, for a 4,2,4 or 4,3,2,3,4
            autoencoder, we use the "2" layer (index 1 or 2, respectively).

        sample : bool, optional
            If True, then draw a sample using the hidden activations as
            independent Bernoulli probabilities for the encoded data. This
            assumes the hidden layer has a logistic sigmoid activation function.

        Returns
        -------
        ndarray :
            The given dataset, encoded by the appropriate hidden layer
            activation.
        '''
        enc = self.feed_forward(x)[(layer or len(self.layers) // 2) - 1]
        if sample:
            return np.random.binomial(n=1, p=enc).astype(np.uint8)
        return enc

    def decode(self, z, layer=None):
        '''Decode an encoded dataset by computing the output layer activation.

        Parameters
        ----------
        z : ndarray
            A matrix containing encoded data from this autoencoder.

        layer : int, optional
            The index of the hidden layer that was used to encode `z`.

        Returns
        -------
        ndarray :
            The decoded dataset.
        '''
        if not hasattr(self, '_decoders'):
            self._decoders = {}
        layer = layer or len(self.layers) // 2
        if layer not in self._decoders:
            self._decoders[layer] = theano.function(
                [self.hiddens[layer - 1]], [self.y], updates=self.updates)
        return self._decoders[layer](z)[0]


class Regressor(Network):
    '''A regressor attempts to produce a target output.'''

    def __init__(self, *args, **kwargs):
        self.k = TT.matrix('k')
        self.k.tag.test_value = np.random.rand(self.batch_size)
        super(Regressor, self).__init__(*args, **kwargs)

    @property
    def inputs(self):
        return [self.x, self.k]

    @property
    def cost(self):
        err = self.y - self.k
        return TT.mean((err * err).sum(axis=1))


class Classifier(Network):
    '''A classifier attempts to match a 1-hot target output.'''

    def __init__(self, *args, **kwargs):
        self.k = TT.ivector('k')
        super(Classifier, self).__init__(*args, **kwargs)
        self.y = self.softmax(self.y)

    @staticmethod
    def softmax(x):
        # TT.nnet.softmax doesn't work with the HF trainer.
        z = TT.exp(x.T - x.T.max(axis=0))
        return (z / z.sum(axis=0)).T

    @property
    def inputs(self):
        return [self.x, self.k]

    @property
    def cost(self):
        return -TT.mean(TT.log(self.y)[TT.arange(self.k.shape[0]), self.k])

    @property
    def incorrect(self):
        return TT.mean(TT.neq(TT.argmax(self.y, axis=1), self.k))

    @property
    def monitors(self):
        yield 'incorrect', self.incorrect
        for i, h in enumerate(self.hiddens):
            yield 'h{}<0.1'.format(i+1), (abs(h) < 0.1).mean()
            yield 'h{}<0.9'.format(i+1), (abs(h) < 0.9).mean()

    def classify(self, x):
        return self.predict(x).argmax(axis=1)
