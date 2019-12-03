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


class TwentyNews(Data):
    def __init__(self, batch_size, data_path='', ctx=None, saveto='', **kwargs):
        self.saveto = saveto
        super(TwentyNews, self).__init__(batch_size, data_path, ctx)

    def load(self, path='~/20news_sklearn', features='BoW', match_avitm=True):
        if path[:2] == '~/':
            path = os.path.join(os.path.expanduser(path[:2]), path[2:])

        ### Specify the file locations
        train_path = path + '/train_sklearn_avitm.npy'
        train_labels_path = path + '/train_labels_sklearn_avitm.npy'
        test_path = path + '/test_sklearn_avitm.npy'
        test_labels_path = path + '/test_labels_sklearn_avitm.npy'
        vocab_path = path + '/vocab.txt'
        label_names_path = path + '/label_names.txt'

        ### Load train
        train = np.load(train_path).astype('float32')
        if train_labels_path:
            train_labels = np.load(train_labels_path)
        else:
            train_labels = None

        ### Load train
        test = np.load(test_path).astype('float32')
        if test_labels_path:
            test_labels = np.load(test_labels_path)
        else:
            test_labels = None

        ### load vocab
        ENCODING = "ISO-8859-1"
        # ENCODING = "utf-8"
        with open(vocab_path, encoding=ENCODING) as f:
             vocab_list = [line.strip('\n') for line in f]

        ### Load label names
        if label_names_path:
            with open(label_names_path, encoding=ENCODING) as f:
                label_name_list = [line.strip('\n') for line in f]
        else:
            label_name_list = None

        # construct maps
        vocab2dim = dict(zip(vocab_list, range(len(vocab_list))))
        dim2vocab = reverse_dict(vocab2dim)

        topic2dim = dict(zip(label_name_list, range(len(label_name_list))))
        dim2topic = reverse_dict(topic2dim)

        return [train, None, test, None, None, None], [train_labels, None, test_labels], [vocab2dim, dim2vocab, topic2dim, dim2topic]