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
import shutil

import numpy as np

from core import Data
from utils import reverse_dict
import scipy.sparse as sparse
import nltk


class Wikitext103(Data):
    def __init__(self, batch_size, data_path='', ctx=None, saveto='', **kwargs):
        self.saveto = saveto
        super(Wikitext103, self).__init__(batch_size, data_path, ctx)

    def load(self, path='./data/wikitext-103', features='BoW', match_avitm=True):
        if path[:2] == '~/':
            path = os.path.join(os.path.expanduser(path[:2]), path[2:])

        ### Specify the file locations
        train_path = path + '/wikitext-103_tra.csr.npz'
        test_path = path + '/wikitext-103_test.csr.npz'
        vocab_path = path + '/vocab.txt'

        ### Load train
        train_csr = sparse.load_npz(train_path)
        train = np.array(train_csr.todense()).astype('float32')

        ### Load test
        test_csr = sparse.load_npz(test_path)
        test = np.array(test_csr.todense()).astype('float32')

        ### load vocab
        ENCODING = "ISO-8859-1"
        # ENCODING = "utf-8"
        with open(vocab_path, encoding=ENCODING) as f:
             vocab_list = [line.strip('\n') for line in f]

        # construct maps
        vocab2dim = dict(zip(vocab_list, range(len(vocab_list))))
        dim2vocab = reverse_dict(vocab2dim)

        return [train, None, test, None, None, None], [None, None, None], [vocab2dim, dim2vocab, None, None]


if __name__ == '__main__':

    def check_create_dir(dir):
        if os.path.exists(dir):  # cleanup existing data folder
            shutil.rmtree(dir)
        os.mkdir(dir)

    # create directory for data
    dataset = 'wikitext-103'
    current_dir = os.getcwd()

    data_dir = os.path.join(current_dir, "data")
    if not os.path.exists(data_dir):
        print('Creating directory:', data_dir)
        os.mkdir(data_dir)
    data_dir = os.path.join(current_dir, "data", dataset)
    check_create_dir(data_dir)
    os.chdir(data_dir)
    print('Current directory: ', os.getcwd())

    # download data
    os.system("curl -O https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip")
    os.system("unzip wikitext-103-v1.zip")

    # parse into documents
    def is_document_start(line):
        if len(line) < 4:
            return False
        if line[0] is '=' and line[-1] is '=':
            if line[2] is not '=':
                return True
            else:
                return False
        else:
            return False


    def token_list_per_doc(input_dir, token_file):
        lines_list = []
        line_prev = ''
        prev_line_start_doc = False
        with open(os.path.join(input_dir, token_file), 'r', encoding='utf-8') as f:
            for l in f:
                line = l.strip()
                if prev_line_start_doc and line:
                    # the previous line should not have been start of a document!
                    lines_list.pop()
                    lines_list[-1] = lines_list[-1] + ' ' + line_prev

                if line:
                    if is_document_start(line) and not line_prev:
                        lines_list.append(line)
                        prev_line_start_doc = True
                    else:
                        lines_list[-1] = lines_list[-1] + ' ' + line
                        prev_line_start_doc = False
                else:
                    prev_line_start_doc = False
                line_prev = line

        print("{} documents parsed!".format(len(lines_list)))
        return lines_list


    input_dir = os.path.join(data_dir, dataset)
    train_file = 'wiki.train.tokens'
    val_file = 'wiki.valid.tokens'
    test_file = 'wiki.test.tokens'
    train_doc_list = token_list_per_doc(input_dir, train_file)
    val_doc_list = token_list_per_doc(input_dir, val_file)
    test_doc_list = token_list_per_doc(input_dir, test_file)

    nltk.download('wordnet')
    from nltk.stem import WordNetLemmatizer
    import re

    token_pattern = re.compile(r"(?u)\b\w\w+\b")
    class LemmaTokenizer(object):
        def __init__(self):
            self.wnl = WordNetLemmatizer()

        def __call__(self, doc):
            return [self.wnl.lemmatize(t) for t in doc.split() if len(t) >= 2 and re.match("[a-z].*", t)
                    and re.match(token_pattern, t)]


    import time
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer

    print('Lemmatizing and counting, this may take a few minutes...')
    start_time = time.time()
    vectorizer = CountVectorizer(input='content', analyzer='word', stop_words='english',
                                 tokenizer=LemmaTokenizer(), max_df=0.8, min_df=3, max_features=20000)

    train_vectors = vectorizer.fit_transform(train_doc_list)
    val_vectors = vectorizer.transform(val_doc_list)
    test_vectors = vectorizer.transform(test_doc_list)

    vocab_list = vectorizer.get_feature_names()
    vocab_size = len(vocab_list)
    print('vocab size:', vocab_size)
    print('Done. Time elapsed: {:.2f}s'.format(time.time() - start_time))

    import scipy.sparse as sparse
    def shuffle_and_dtype(vectors):
        idx = np.arange(vectors.shape[0])
        np.random.shuffle(idx)
        vectors = vectors[idx]
        vectors = sparse.csr_matrix(vectors, dtype=np.float32)
        print(type(vectors), vectors.dtype)
        return vectors

    train_vectors = shuffle_and_dtype(train_vectors)
    val_vectors = shuffle_and_dtype(val_vectors)
    test_vectors = shuffle_and_dtype(test_vectors)

    with open('vocab.txt', 'w', encoding='utf-8') as f:
        for item in vocab_list:
            f.write(item+'\n')

    sparse.save_npz('wikitext-103_tra.csr.npz', train_vectors)
    sparse.save_npz('wikitext-103_val.csr.npz', val_vectors)
    sparse.save_npz('wikitext-103_test.csr.npz', test_vectors)
