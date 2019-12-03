# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import numpy as np
from scipy.special import logit, expit

import mxnet as mx
from mxnet.gluon import nn

from core import ENet, DNet


class Encoder(ENet):
    '''
    A gluon HybridBlock Encoder class
    '''
    def __init__(self, model_ctx, batch_size, input_dim, n_hidden=64, ndim_y=16, ndim_z=10, n_layers=0, nonlin=None,
                 weights_file='', freeze=False, latent_nonlin='sigmoid', **kwargs):
        '''
        Constructor for encoder.

        Args
        ----
        model_ctx: mxnet device context, No default
          Which device to store/run the data and model on.
        batch_size: integer, No default
          The minibatch size.
        input_dim: integer, No default
          The data dimensionality that is input to the encoder.
        n_hidden: integer or list, default 64
          If integer, specifies the number of hidden units in
          every hidden layer.
          If list, each element specifies the number of hidden
          units in each hidden layer.
        output_dim: integer, default 10
          The dimensionality of the latent space, z.
        n_layers: integer, default 0
          The number of hidden layers.
        nonlin: string, default None
          The nonlinearity to use in every hidden layer.
        weights_file: string, default ''
          The path to the file (mxnet params file or pickle file)
          containing weights for each layer of the encoder.
        freeze: boolean, default False
          Whether to freeze the encoder weights (MIGHT BE BROKEN).
        latent_nonlin: string, default 'sigmoid'
          Which space to use for the latent variable:
            if 'sigmoid': z in (0,1)
            else: z in (-inf,inf)

        Parameters
        ----------
        Returns
        -------
        encoder object.
        '''
        super(Encoder, self).__init__()

        if n_layers >= 0:
            if isinstance(n_hidden, list):
                n_hidden = n_hidden[0]
                print('NOTE: Encoder ignoring list of hiddens because n_layer >= 0. Just using first element.')
            n_hidden = n_layers*[n_hidden]
        else:
            n_layers = len(n_hidden)
            print('NOTE: Encoder reading n_hidden as list.')

        if nonlin == '':
            nonlin = None
        
        in_units = input_dim
        with self.name_scope():
            self.main = nn.HybridSequential(prefix='encoder')
            for i in range(n_layers):
                self.main.add(nn.Dense(n_hidden[i], in_units=in_units, activation=nonlin))
                in_units = n_hidden[i]
            self.main.add(nn.Dense(ndim_y, in_units=in_units, activation=None))

        self.model_ctx = model_ctx
        self.input_dim = input_dim
        self.n_hidden = n_hidden
        self.ndim_y = ndim_y
        self.ndim_z = ndim_z
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.nonlin = nonlin
        self.latent_nonlin = latent_nonlin
        self.weights_file = weights_file
        self.freeze = freeze
        self.dist_params = [None]

    def hybrid_forward(self, F, x):
        '''
        Passes the input through the encoder.

        Args
        ----
        F: mxnet.nd or mxnet.sym, No default
          This will be passed implicitly when calling hybrid forward.
        x: NDarray or mxnet symbol, No default
          The input to the encoder.

        Returns
        -------
        dist_params: list
          A list of the posterior parameters as NDarrays, each being of size batch_size x z_dim.
        samples: NDarray
          The posterior samples as a batch_size x z_dim NDarray.
        '''

        y = self.main(x)

        return y


class Decoder(DNet):
    '''
    A gluon HybridBlock Decoder class with Multinomial likelihood, p(x|z).
    '''
    def __init__(self, model_ctx, batch_size, output_dim, ndim_y=16,  n_hidden=64, n_layers=0, nonlin='',
                 weights_file='', freeze=False, latent_nonlin='', **kwargs):
        '''
        Constructor for Multinomial decoder.

        Args
        ----
        model_ctx: mxnet device context, No default
          Which device to store/run the data and model on.
        batch_size: integer, No default
          The minibatch size.
        n_hidden: integer or list, default 64
          If integer, specifies the number of hidden units in
          every hidden layer.
          If list, each element specifies the number of hidden
          units in each hidden layer.
        output_dim: integer, No default
          The dimensionality of the latent space, z.
        n_layers: integer, default 0
          The number of hidden layers.
        nonlin: string, default 'sigmoid'
          The nonlinearity to use in every hidden layer.
        weights_file: string, default ''
          The path to the file (mxnet params file or pickle file)
          containing weights for each layer of the encoder.
        freeze: boolean, default False
          Whether to freeze the encoder weights (MIGHT BE BROKEN).
        latent_nonlin: string, default 'sigmoid'
          Which space to use for the latent variable:
            if 'sigmoid': z in (0,1)
            else: z in (-inf,inf)

        Parameters
        ----------
        Returns
        -------
        Multinomial decoder object.
        '''
        super(Decoder, self).__init__()

        if n_layers >= 0:
            if isinstance(n_hidden, list):
                n_hidden = n_hidden[0]
                print('NOTE: Decoder ignoring list of hiddens because n_layer >= 0. Just using first element.')
            n_hidden = n_layers*[n_hidden]
        else:
            n_layers = len(n_hidden)
            print('NOTE: Decoder reading n_hidden as list.')

        if nonlin == '':
            nonlin = None

        in_units = n_hidden[0]
        with self.name_scope():
            self.main = nn.HybridSequential(prefix='decoder')
            self.main.add(nn.Dense(n_hidden[0], in_units=ndim_y, activation=None))

        self.model_ctx = model_ctx
        self.batch_size = batch_size
        self.ndim_y = ndim_y
        self.n_hidden = n_hidden
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.nonlin = nonlin
        self.latent_nonlin = latent_nonlin
        self.weights_file = weights_file
        self.freeze = freeze

    def hybrid_forward(self, F, y):
        '''
        Passes the input through the decoder.

        Args
        ----
        F: mxnet.nd or mxnet.sym, No default
          This will be passed implicitly when calling hybrid forward.
        x: NDarray or mxnet symbol, No default
          The input to the decoder.

        Returns
        -------
        dist_params: list
          A list of the multinomial parameters as NDarrays, each being of size batch_size x z_dim.
        samples: NDarray
          The multinomial samples as a batch_size x z_dim NDarray (NOT IMPLEMENTED).
        '''
        out = self.main(y)
        return out

    def y_as_topics(self, eps=1e-10):
        y = np.eye(self.ndim_y)
        return mx.nd.array(y)

class Discriminator_y(ENet):
    '''
    A gluon HybridBlock Discriminator Class for y
    '''

    def __init__(self, model_ctx, batch_size, output_dim=2, ndim_y=16, n_hidden=64, n_layers=0, nonlin='sigmoid',
                 weights_file='', freeze=False, latent_nonlin='sigmoid', apply_softmax=False, **kwargs):
        '''
        Constructor for Discriminator Class for y.

        Args
        ----
        model_ctx: mxnet device context, No default
          Which device to store/run the data and model on.
        batch_size: integer, No default
          The minibatch size.
        n_hidden: integer or list, default 64
          If integer, specifies the number of hidden units in
          every hidden layer.
          If list, each element specifies the number of hidden
          units in each hidden layer.
        output_dim: integer, No default
          The dimensionality of the latent space, z.
        n_layers: integer, default 0
          The number of hidden layers.
        nonlin: string, default 'sigmoid'
          The nonlinearity to use in every hidden layer.
        weights_file: string, default ''
          The path to the file (mxnet params file or pickle file)
          containing weights for each layer of the encoder.
        freeze: boolean, default False
          Whether to freeze the encoder weights (MIGHT BE BROKEN).
        latent_nonlin: string, default 'sigmoid'
          Which space to use for the latent variable:
            if 'sigmoid': z in (0,1)
            else: z in (-inf,inf)

        Parameters
        ----------
        Returns
        -------
        Multinomial Discriminator object.
        '''
        super(Discriminator_y, self).__init__()

        if n_layers >= 0:
            if isinstance(n_hidden, list):
                n_hidden = n_hidden[0]
                print('NOTE: Decoder ignoring list of hiddens because n_layer >= 0. Just using first element.')
            n_hidden = n_layers * [n_hidden]
        else:
            n_layers = len(n_hidden)
            print('NOTE: Decoder reading n_hidden as list.')

        if latent_nonlin != 'sigmoid':
            print('NOTE: Latent z will be fed to decoder in logit-space (-inf,inf).')
        else:
            print('NOTE: Latent z will be fed to decoder in probability-space (0,1).')

        if nonlin == '':
            nonlin = None

        in_units = ndim_y
        with self.name_scope():
            self.main = nn.HybridSequential(prefix='discriminator_y')
            for i in range(n_layers):
                self.main.add(nn.Dense(n_hidden[i], in_units=in_units, activation=nonlin))
                in_units = n_hidden[i]
            self.main.add(nn.Dense(output_dim, in_units=in_units, activation=None))

        self.model_ctx = model_ctx
        self.batch_size = batch_size
        self.ndim_y = ndim_y
        self.n_hidden = n_hidden
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.nonlin = nonlin
        self.latent_nonlin = latent_nonlin
        self.weights_file = weights_file
        self.freeze = freeze
        self.apply_softmax = apply_softmax

    def hybrid_forward(self, F, y):
        '''
        Passes the input through the decoder.

        Args
        ----
        F: mxnet.nd or mxnet.sym, No default
          This will be passed implicitly when calling hybrid forward.
        x: NDarray or mxnet symbol, No default
          The input to the decoder.
        '''
        logit = self.main(y)
        if self.apply_softmax:
            return F.softmax(logit)
        return logit
