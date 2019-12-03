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

import pickle
import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import log_loss, v_measure_score

import mxnet as mx
from mxnet import gluon, io


# import misc as nm
# import datasets as nuds
import scipy.sparse as sparse
import json
# import wordvectors as nuwe


class Data(object):
    '''
    Data Generator object. Main functionality is contained in ``minibatch'' method
    and ``subsampled_labeled_data'' if training in a semi-supervised fashion.
    Introducing new datasets requires implementing ``load'' and possibly overwriting
    portions of ``__init__''.
    '''
    def __init__(self, batch_size=1, data_path='', ctx=mx.cpu(0)):
        '''
        Constructor for Data.

        Args
        ----
        batch_size: int, default 1
          An integer specifying the batch size - required for precompiling the graph.
        data_path: string, default ''
          This is primarily used by Mulan to specify which dataset to load from Mulan,
          e.g., data_path='bibtex'.
        ctx: mxnet device context, default mx.cpu(0)
          Which device to store/run the data and model on.

        Returns
        -------
        Data object
        '''
        self.batch_size = batch_size
        if data_path == '':
            data, labels, maps = self.load()
        else:
            data, labels, maps = self.load(data_path)
        self.ctx = ctx
        # # normalize the data:
        # def softmax(x):
        #     """Compute softmax values for each sets of scores in x."""
        #     e_x = np.exp(x - np.max(x, axis=1).reshape((-1,1)))
        #     return e_x / np.sum(e_x, axis=1).reshape((-1,1))
        # for i in range(len(data)):
        #     data[i] = softmax(data[i])

        data_names = ['train','valid','test','train_with_labels','valid_with_labels','test_with_labels']
        label_names = ['train_label', 'valid_label', 'test_label']

        self.data = dict(zip(data_names, data))
        self.labels = dict(zip(label_names, labels))

        # repeat data to at least match batch_size
        for k, v in self.data.items():
            if v is not None and v.shape[0] < self.batch_size:
                print('NOTE: Number of samples for {0} is smaller than batch_size ({1}<{2}). Duplicating samples to exceed batch_size.'.format(k,v.shape[0],self.batch_size))
                if type(v) is np.ndarray:
                    self.data[k] = np.tile(v, (self.batch_size // v.shape[0] + 1, 1))
                else:
                    self.data[k] = mx.nd.tile(v, (self.batch_size // v.shape[0] + 1, 1))

        for k, v in self.labels.items():
            if v is not None and v.shape[0] < self.batch_size:
                print('NOTE: Number of samples for {0} is smaller than batch_size ({1}<{2}). Duplicating samples to exceed batch_size.'.format(k,v.shape[0],self.batch_size))
                self.labels[k] = np.tile(v, (self.batch_size // v.shape[0] + 1, ))

        map_names = ['vocab2dim','dim2vocab','topic2dim','dim2topic']
        self.maps = dict(zip(map_names, maps))
        dls = [self.dataloader(d, batch_size) for d in data]
        dis = [iter(dl) if dl is not None else None for dl in dls]
        self.dataloaders = dict(zip(data_names, dls))
        self.dataiters = dict(zip(data_names, dis))
        self.wasreset = dict(zip(data_names, np.ones(len(data_names), dtype='bool')))

        self.data_dim = self.data['train'].shape[1]
        if self.data['train_with_labels'] is not None:
            self.label_dim = self.data['train_with_labels'].shape[1] - self.data['train'].shape[1]


    def dataloader(self, data, batch_size, shuffle=True):
        '''
        Constructs a data loader for generating minibatches of data.

        Args
        ----
        data: numpy array, no default
          The data from which to load minibatches.
        batch_size: integer, no default
          The # of samples returned in each minibatch.
        shuffle: boolean, default True
          Whether or not to shuffle the data prior to returning the data loader.

        Returns
        -------
        DataLoader: A gluon DataLoader iterator
        '''
        if data is None:
            return None
        else:
            # inds = np.arange(data.shape[0])
            # if shuffle:
            #     np.random.shuffle(inds)
            # ordered = data[inds]
            # N, r = divmod(data.shape[0], batch_size)
            # if r > 0:
            #     ordered = np.vstack([ordered, ordered[:r]])
            if type(data) is np.ndarray:
                return gluon.data.DataLoader(data, batch_size, last_batch='discard', shuffle=shuffle)
            else:
                return io.NDArrayIter(data={'data': data}, batch_size=batch_size, shuffle=shuffle, last_batch_handle='discard')

    def force_reset_data(self, key, shuffle=True):
        '''
        Resets minibatch index to zero to restart an epoch.

        Args
        ----
        key: string, no default
          Required to select appropriate data in ``data'' object,
          e.g., 'train', 'test', 'train_with_labels', 'test_with_labels'.
        shuffle: boolean, default True
          Whether or not to shuffle the data prior to returning the data loader.

        Returns
        -------
        Nothing.
        '''
        if self.data[key] is not None:
            if type(self.data[key]) is np.ndarray:
                self.dataloaders[key] = self.dataloader(self.data[key], self.batch_size, shuffle)
                self.dataiters[key] = iter(self.dataloaders[key])
            else:
                self.dataiters[key].hard_reset()
            self.wasreset[key] = True

    def minibatch(self, key, pad_width=0):
        '''
        Returns a minibatch of data (stored on device self.ctx).

        Args
        ----
        key: string, no default
          Required to select appropriate data in ``data'' object,
          e.g., 'train', 'test', 'train_with_labels', 'test_with_labels'.
        pad_width: integer, default 0
          The amount to zero-pad the labels to match the dimensionality of z.

        Returns
        -------
        minibatch: NDArray on device self.ctx
          An NDArray of size batch_size x # of features.
        '''
        if self.dataiters[key] is None:
            return None
        else:
            if type(self.data[key]) is np.ndarray:
                try:
                    mb = self.dataiters[key].__next__().reshape((self.batch_size, -1))
                    if pad_width > 0:
                        mb = mx.nd.concat(mb, mx.nd.zeros((self.batch_size, pad_width)))
                    return mb.copyto(self.ctx)
                except:
                    self.force_reset_data(key)
                    mb = self.dataiters[key].__next__().reshape((self.batch_size, -1))
                    if pad_width > 0:
                        mb = mx.nd.concat(mb, mx.nd.zeros((self.batch_size, pad_width)))
                    return mb.copyto(self.ctx)
            else:
                try:
                    mb = self.dataiters[key].__next__().data[0].as_in_context(self.ctx)
                    return mb
                except:
                    self.dataiters[key].hard_reset()
                    mb = self.dataiters[key].__next__().data[0].as_in_context(self.ctx)
                    return mb

    def get_documents(self, key, split_on=None):
        '''
        Retrieves a minibatch of documents via ``data'' object parameter.

        Args
        ----
        key: string, no default
          Required to select appropriate data in ``data'' object,
          e.g., 'train', 'test', 'train_with_labels', 'test_with_labels'.
        split_on: integer, default None
          Useful if self.data[key] contains both data and labels in one
          matrix and want to split them, e.g., split_on = data_dim.

        Returns
        -------
        minibatch: NDArray if split_on is None, else [NDarray, NDArray]
        '''
        if 'labels' in key:
            batch = self.minibatch(key, pad_width=self.label_pad_width)
        else:
            batch = self.minibatch(key)
        if split_on is not None:
            batch, labels = batch[:,:split_on], batch[:,split_on:]
            return batch, labels
        else:
            return batch

    @staticmethod
    def visualize_series(y, ylabel, file, args, iteration, total_samples, labels=None):
        '''
        Plots and saves a figure of y vs iterations and epochs to file.

        Args
        ----
        y: a list (of lists) or numpy array, no default
          A list (of possibly another list) of numbers to plot.
        ylabel: string, no default
          The label for the y-axis.
        file: string, no default
          A path with filename to save the figure to.
        args: dictionary, no default
          A dictionary of model, training, and evaluation specifications.
        iteration: integer, no default
          The current iteration in training.
        total_samples: integer, no default
          The total number of samples in the dataset - used along with batch_size
          to convert iterations to epochs.
        labels: list of strings, default None
          If y is a list of lists, the labels contains names for each element
          in the nested list. This is used to create an appropriate legend
          for the plot.

        Returns
        -------
        Nothing.
        '''
        if len(y) > 0:
            fig = plt.figure()
            ax = plt.subplot(111)
            x = np.linspace(0, iteration, num=len(y)) * args['batch_size'] / total_samples
            y = np.array(y)
            if len(y.shape) > 1:
                for i in range(y.shape[1]):
                    if labels is None:
                        plt.plot(x,y[:,i])
                    else:
                        plt.plot(x,y[:,i], label=labels[i])
            else:
                plt.plot(x,y)
            ax.set_ylabel(ylabel)
            ax.set_xlabel('Epochs')
            plt.grid(True)

            ax2 = ax.twiny()

            # https://pythonmatplotlibtips.blogspot.com/2018/01/add-second-x-axis-below-first-x-axis-python-matplotlib-pyplot.html
            # Decide the ticklabel position in the new x-axis,
            # then convert them to the position in the old x-axis
            # xticks list seems to be padded with extra lower and upper ticks --> subtract 2 from length
            newlabel = np.around(np.linspace(0, iteration, num=len(ax.get_xticks())-2)).astype('int') # labels of the xticklabels: the position in the new x-axis
            # ax2.set_xticks(ax.get_xticks())
            ax2.set_xticks(newlabel * args['batch_size'] / total_samples)
            ax2.set_xticklabels(newlabel//1000)

            ax2.xaxis.set_ticks_position('bottom') # set the position of the second x-axis to bottom
            ax2.xaxis.set_label_position('bottom') # set the position of the second x-axis to bottom
            ax2.spines['bottom'].set_position(('outward', 36))
            ax2.set_xlabel('Thousand Iterations')
            ax2.set_xlim(ax.get_xlim())

            if labels is not None:
                lgd = ax.legend(loc='center left', bbox_to_anchor=(1.05, 1))
                fig.savefig(args['saveto']+file, additional_artists=[lgd], bbox_inches='tight')
            else:
                fig.tight_layout()
                fig.savefig(args['saveto']+file)
            plt.close()

    def load(self, path=''):
        '''
        Loads data and maps from path.

        Args
        ----
        path: string, default ''
          A path to the data file.

        Returns
        -------
        data: list of numpy arrays
          A list of the different subsets of data, e.g.,
          `train', `test', 'train_with_labels', 'test_with_labels'.
        maps: list of dictionaries
          A list of dictionaries for mapping between dimensions and strings,
          e.g., 'vocab2dim', 'dim2vocab', 'topic2dim', 'dim2topic'.
        '''
        data = [np.empty((1,1)) for data in ['train','valid','test','train_with_labels','valid_with_labels','test_with_labels']]
        maps = [{'a':0}, {0:'a'}, {'Letters':0}, {0:'Letters'}]
        self.data_path = path + '***.npz'
        return data, maps


class ENet(gluon.HybridBlock):
    '''
    A gluon HybridBlock Encoder (skeleton) class.
    '''
    def __init__(self):
        '''
        Constructor for Encoder.

        Args
        ----
        None

        Returns
        -------
        Encoder object
        '''
        super(ENet, self).__init__()
            
    def hybrid_forward(self, x):
        '''
        Encodes x.

        Args
        ----
        x: mx.NDArray or sym, No default
          Input to encoder.

        Returns (should)
        -------
        params: list of NDArray or sym
          parameters for the encoding distribution
        samples: NDArray or sym
          samples drawn from the encoding distribution
        '''
        raise NotImplementedError('Need to write your own Encoder that inherits from ENet. Put this file in models/.')

    def init_weights(self, weights=None):
        '''
        Initializes the encoder weights. Default is Xavier initialization.

        Args
        ----
        weights: list of numpy arrays, No default
          Weights to load into the model. Not required. Preference is to
          load weights from file.

        Returns
        -------
        Nothing.
        '''
        loaded = False
        source = 'keyword argument'
        if self.weights_file != '' and weights is None:
            try:
                self.load_params(self.weights_file, self.model_ctx)
                source = 'mxnet weights file: '+self.weights_file
                print('NOTE: Loaded encoder weights from '+source+'.')
                if self.freeze:
                    self.freeze_params()
                    print('NOTE: Froze encoder weights from '+source+'.')
                weights = None
                loaded = True
            except:
                weights = pickle.load(open(self.weights_file,'rb'))
                source = 'pickle file: '+self.weights_file
        if weights is not None:
            assert self.n_layers == 0
            for p,w in zip(self.collect_params().values(), weights):
                if w is not None:
                    p.initialize(mx.init.Constant(mx.nd.array(w.squeeze())), ctx=self.model_ctx)
                    if self.freeze:
                        p.lr_mult = 0.
            print('NOTE: Loaded encoder weights from '+source+'.')
            if self.freeze:
                print('NOTE: Froze encoder weights from '+source+'.')
            loaded = True
        if not loaded:
            self.collect_params().initialize(mx.init.Xavier(), ctx=self.model_ctx)
            print('NOTE: Randomly initialized encoder weights.')
            # self.collect_params().initialize(mx.init.Zero(), ctx=self.model_ctx)
            # print('NOTE: initialized encoder weights to ZERO.')

    def freeze_params(self):
        for p in self.collect_params().values():
            p.lr_mult = 0.


class DNet(gluon.HybridBlock):
    '''
    A gluon HybridBlock Decoder (skeleton) class.
    '''
    def __init__(self):
        '''
        Constructor for Decoder.

        Args
        ----
        None

        Returns
        -------
        Decoder object
        '''
        super(DNet, self).__init__()

    def hybrid_forward(self, y, z):
        '''
        Decodes x.

        Args
        ----
        x: mx.NDArray or sym, no default
          Input to decoder.

        Returns (should)
        -------
        params: list of NDArray or sym
          parameters for the encoding distribution
        samples: NDArray or sym
          samples drawn from the encoding distribution. None if sampling is not implemented.
        '''
        raise NotImplementedError('Need to write your own Decoder that inherits from ENet. Put this file in models/.')

    def init_weights(self, weights=None):
        '''
        Initializes the decoder weights. Default is Xavier initialization.

        Args
        ----
        weights: list of numpy arrays, No default
          Weights to load into the model. Not required. Preference is to
          load weights from file.

        Returns
        -------
        Nothing.
        '''
        loaded = False
        source = 'keyword argument'
        if self.weights_file != '' and weights is None:
            try:
                self.load_params(self.weights_file, self.model_ctx)
                source = 'mxnet weights file: '+self.weights_file
                print('NOTE: Loaded decoder weights from '+source+'.')
                if self.freeze:
                    self.freeze_params()
                    print('NOTE: Froze decoder weights from '+source+'.')
                weights = None
                loaded = True
            except:
                weights = pickle.load(open(self.weights_file,'rb'))
                source = 'pickle file: '+self.weights_file
        if weights is not None:
            assert self.n_layers == 0
            for p,w in zip(self.collect_params().values(), weights):
                if w is not None:
                    p.initialize(mx.init.Constant(mx.nd.array(w.squeeze())), ctx=self.model_ctx)
                    if self.freeze:
                        p.lr_mult = 0.
            print('NOTE: Loaded decoder weights from '+source+'.')
            if self.freeze:
                print('NOTE: Froze decoder weights from '+source+'.')
            loaded = True
        if not loaded:
            self.collect_params().initialize(mx.init.Xavier(), ctx=self.model_ctx)
            print('NOTE: Randomly initialized decoder weights.')

    def freeze_params(self):
        for p in self.collect_params().values():
            p.lr_mult = 0.


class Compute(object):
    '''
    Skeleton class to manage training, testing, and retrieving outputs.
    See ``compute_op.py'' for ``flesh''.
    '''
    def __init__(self,  data, Enc, Dec,  Dis_y, args):
        '''
        Constructor for Compute.

        Returns
        -------
        Compute object
        '''
        self.data = data
        self.Enc = Enc
        self.Dec = Dec
        self.Dis_y = Dis_y
        self.args = args
        self.model_ctx = Enc.model_ctx
        self.ndim_y = args['ndim_y']

        weights_enc = Enc.collect_params()
        weights_dec = Dec.collect_params()
        weights_dis_y = Dis_y.collect_params()

        if self.args['optim'] == 'Adam':
            # args_dict = {'learning_rate': self.args['learning_rate'], 'beta1': self.args['betas'][0], 'beta2': self.args['betas'][1], 'epsilon': self.args['epsilon']}
            # optimizer_enc = gluon.Trainer(weights_enc, 'adam', args_dict)
            # optimizer_dec = gluon.Trainer(weights_dec, 'adam', args_dict)
            # optimizer_dis_y = gluon.Trainer(weights_dis_y, 'adam', args_dict)
            optimizer_enc = gluon.Trainer(weights_enc, 'adam', {'learning_rate': self.args['learning_rate'], 'beta1': 0.99})
            optimizer_dec = gluon.Trainer(weights_dec, 'adam', {'learning_rate': self.args['learning_rate'], 'beta1': 0.99})
            optimizer_dis_y = gluon.Trainer(weights_dis_y, 'adam', {'learning_rate': self.args['learning_rate']})
        if self.args['optim'] == 'Adadelta':
            # note: learning rate has no effect on Adadelta --> https://mxnet.incubator.apache.org/_modules/mxnet/optimizer.html#AdaDelta
            args_dict = {'rescale_grad': 1}  #, 'clip_gradient': 0.1}
            optimizer_enc = gluon.Trainer(weights_enc, 'adadelta', args_dict)
            optimizer_dec = gluon.Trainer(weights_dec, 'adadelta', args_dict)
            optimizer_dis_y = gluon.Trainer(weights_dis_y, 'adadelta', args_dict)
        if self.args['optim'] == 'RMSprop':
            args_dict = {'learning_rate': self.args['learning_rate'], 'epsilon': 1e-10, 'alpha': 0.9}
            optimizer_enc = gluon.Trainer(weights_enc, 'rmsprop', args_dict)
            optimizer_dec = gluon.Trainer(weights_dec, 'rmsprop', args_dict)
            optimizer_dis_y = gluon.Trainer(weights_dis_y, 'rmsprop', args_dict)
        if self.args['optim'] == 'SGD':
            args_dict = {'learning_rate': self.args['learning_rate'], 'wd': self.args['weight_decay'], 'rescale_grad': 1., 'momentum': 0.0, 'lazy_update': False}
            optimizer_enc = gluon.Trainer(weights_enc, 'sgd', args_dict)
            optimizer_dec = gluon.Trainer(weights_dec, 'sgd', args_dict)
            optimizer_dis_y = gluon.Trainer(weights_dis_y, 'sgd', args_dict)

        self.optimizer_enc = optimizer_enc
        self.optimizer_dec = optimizer_dec
        self.optimizer_dis_y = optimizer_dis_y
        self.weights_enc = weights_enc
        self.weights_dec = weights_dec
        self.weights_dis_y = weights_dis_y

    def train_op(self):
        '''
        Trains the model using one minibatch of data.
        '''
        return None, None, None, None

    def test_op(self, num_samples=None, num_epochs=None, reset=True, dataset='test'):
        '''
        Evaluates the model using num_samples.

        Args
        ----
        num_samples: integer, default None
          The number of samples to evaluate on. This is converted to
          evaluating on (num_samples // batch_size) minibatches.
        num_epochs: integer, default None
          The number of epochs to evaluate on. This used if num_samples
          is not specified. If neither is specified, defaults to 1 epoch.
        reset: bool, default True
          Whether to reset the test data index to 0 before iterating
          through and evaluating on minibatches.
        dataset: string, default 'test':
          Which dataset to evaluate on: 'valid' or 'test'.
        '''
        if num_samples is None:
            num_samples = self.data.data[dataset].shape[0]

        if reset:
            # Reset Data to Index Zero
            self.data.force_reset_data(dataset)
            self.data.force_reset_data(dataset+'_with_labels')

        return None, None, None, None

    def get_outputs(self, num_samples=None, num_epochs=None, reset=True, dataset='test'):
        '''
        Retrieves raw outputs from model for num_samples.

        Args
        ----
        num_samples: integer, default None
          The number of samples to evaluate on. This is converted to
          evaluating on (num_samples // batch_size) minibatches.
        num_epochs: integer, default None
          The number of epochs to evaluate on. This used if num_samples
          is not specified. If neither is specified, defaults to 1 epoch.
        reset: bool, default True
          Whether to reset the test data index to 0 before iterating
          through and evaluating on minibatches.
        dataset: string, default 'test':
          Which dataset to evaluate on: 'valid' or 'test'.
        '''
        if num_samples is None:
            num_samples = self.data.data[dataset].shape[0]

        if reset:
            # Reset Data to Index Zero
            self.data.force_reset_data(dataset)
            self.data.force_reset_data(dataset+'_with_labels')

        return None, None, None, None, None, None