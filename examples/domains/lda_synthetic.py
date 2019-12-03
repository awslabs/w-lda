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

import os

import numpy as np

from core import Data
from utils import reverse_dict
import scipy.sparse as sparse
import mxnet as mx


class LdaSynthetic(Data):
    def __init__(self, batch_size, data_path='', ctx=None, saveto='', **kwargs):
        self.saveto = saveto
        super(LdaSynthetic, self).__init__(batch_size, data_path, ctx)

    def load(self, path='./lda_synthetic', features='BoW', match_avitm=False):
        if path[:2] == '~/':
            path = os.path.join(os.path.expanduser(path[:2]), path[2:])

        ### Specify the file locations
        train_path = path + '/lda_synthetic_train.npz'
        val_path = path + '/lda_synthetic_train.npz'
        test_path = path + '/lda_synthetic_test.npz'
        vocab_path = path + '/vocab.txt'

        ### Load train
        # train_csr = sparse.load_npz(train_path)
        # train = np.array(train_csr.todense()).astype('float32')
        train = sparse.load_npz(train_path).astype('float32')
        train = mx.nd.sparse.csr_matrix(train, dtype='float32')

        ### Load val
        val = sparse.load_npz(val_path).astype('float32')
        val = mx.nd.sparse.csr_matrix(val, dtype='float32')

        ### Load test
        # test_csr = sparse.load_npz(test_path)
        # test = np.array(test_csr.todense()).astype('float32')
        test = sparse.load_npz(test_path).astype('float32')
        test = mx.nd.sparse.csr_matrix(test, dtype='float32')

        ### load vocab
        # ENCODING = "ISO-8859-1"
        ENCODING = "utf-8"
        with open(vocab_path, encoding=ENCODING) as f:
             vocab_list = [line.strip('\n') for line in f]

        # construct maps
        vocab2dim = dict(zip(vocab_list, range(len(vocab_list))))
        dim2vocab = reverse_dict(vocab2dim)

        return [train, val, test, None, None, None], [None, None, None], [vocab2dim, dim2vocab, None, None]