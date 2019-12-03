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

import sys
import re
import math
import os
import time
import socket
import itertools

py_version = 2
if sys.version_info.major == 3:
    import _pickle as pickle

    py_version = 3
else:
    import cPickle as pickle
# log_prefix = re.compile(r"^\[[^\]]+\]")
# topics_pattern = re.compile(r'[T|t]opics from epoch:([^\s]+)\s*\(num_topics:([0-9]+)\):')

phrase_split_pattern = re.compile(r'-|_')


def get_terminal_width():
    try:
        term_cols = os.get_terminal_size().columns
    except:
        term_cols = 80
    return term_cols


def print_center(string):
    n_cols = get_terminal_width()
    spacer = '  '
    center_string = spacer + string + spacer
    n_front = (n_cols - len(center_string)) // 2
    n_back = n_cols - len(center_string) - n_front
    new_string = ' ' * n_front + center_string + ' ' * n_back
    print(new_string)


def print_header(string, skipline=True, symbol='#', doubleline=False):
    n_cols = get_terminal_width()
    if skipline:
        print()
    print(symbol * n_cols)
    if doubleline:
        print(symbol * n_cols)
    print_center(string)
    print(symbol * n_cols)
    if doubleline:
        print(symbol * n_cols)
    if skipline:
        print()


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    pass_str = OKGREEN + 'Pass' + ENDC
    fail_str = FAIL + 'Fail' + ENDC


def print_warning(*args):
    s = ' '.join(args)
    print(bcolors.WARNING + '{}'.format(s) + bcolors.ENDC)


def print_green(*args):
    s = ' '.join(args)
    print(bcolors.OKGREEN + '{}'.format(s) + bcolors.ENDC)


def print_blue(*args):
    s = ' '.join(args)
    print(bcolors.OKBLUE + '{}'.format(s) + bcolors.ENDC)


def print_error(*args):
    s = ' '.join(args)
    print(bcolors.FAIL + '{}'.format(s) + bcolors.ENDC)


class RefCorpus:
    def __init__(self):
        home_dir = os.path.expanduser('~')
        self.wiki_invind_file = os.path.join(home_dir, 'wikipedia.inv_index.pkl')
        self.wiki_dict_file = os.path.join(home_dir, 'wikipedia.dict.pkl')

    def load_corpus(self):
        print_header('Loading reference corpus')

        with open(self.wiki_dict_file, 'rb') as f:
            if py_version == 3:
                corpus_vocab = pickle.load(f, encoding='utf-8')
            else:
                corpus_vocab = pickle.load(f)

        print(len(corpus_vocab))

        with open(self.wiki_invind_file, 'rb') as f:
            [inv_index, corpus_size] = pickle.load(f)

        self.corpus_vocab = corpus_vocab  # number of words
        self.inv_index = inv_index  # word_id: [doc_id1, doc_id2...]
        self.corpus_size = corpus_size  # number of documents


def get_docs_from_index(w, corpus_vocab, inv_index):
    wdocs = set()
    if re.search(phrase_split_pattern, w):
        # this is to handle the phrases in NYT corpus, without which we will have 50% of the words considered OOV.
        wdocs = intersecting_docs(w, corpus_vocab, inv_index)
    elif w in corpus_vocab:
        wdocs = inv_index[corpus_vocab[w]]
    return wdocs


def intersecting_docs(phrase, corpus_vocab, inverted_index):
    words = re.split(phrase_split_pattern, phrase)
    intersect_docs = set()
    for word in words:
        if not word in corpus_vocab:
            # if any of the words in the phrase is not the corpus, the phrase also is not in the corpus
            return set()
        if not intersect_docs:
            intersect_docs.update(inverted_index[corpus_vocab[word]])
        else:
            intersect_docs.intersection_update(inverted_index[corpus_vocab[word]])
    return intersect_docs


def get_pmi(docs_1, docs_2, corpus_size):
    assert len(docs_1)
    assert len(docs_2)
    small, big = (docs_1, docs_2) if len(docs_1) < len(docs_2) else (docs_2, docs_1)
    intersect = small.intersection(big)
    pmi = 0.0
    npmi = 0.0
    if len(intersect):
        pmi = math.log(corpus_size) + math.log(len(intersect)) - math.log(len(docs_1)) - math.log((len(docs_2)))
        npmi = -1 * pmi / (math.log(len(intersect)) - math.log(corpus_size))

    return pmi, npmi


def get_idf(w, inv_index, corpus_vocab, corpus_size):
    n_docs = len(get_docs_from_index(w, corpus_vocab, inv_index))
    return math.log(corpus_size / (n_docs + 1.0))


def test_pmi(inv_index, corpus_vocab, corpus_size):
    word_pairs = [
        ["apple", "ipad"],
        ["monkey", "business"],
        ["white", "house"],
        ["republican", "democrat"],
        ["china", "usa"],
        ["president", "bush"],
        ["president", "george_bush"],
        ["president", "george-bush"]
    ]
    pmis = []
    for pair in word_pairs:
        w1docs = get_docs_from_index(pair[0], corpus_vocab, inv_index)
        w2docs = get_docs_from_index(pair[1], corpus_vocab, inv_index)
        assert len(w1docs)
        assert len(w2docs)
        pmi, _ = get_pmi(w1docs, w2docs, corpus_size)
        assert pmi > 0.0
        print("Testing PMI: w1: {}  w2: {}  pmi: {}".format(pair[0], pair[1], pmi))
        pmis.append(pmi)
    assert pmis[0] > pmis[1]  # pmi(apple, ipad) > pmi(monkey, business)



def get_topic_pmi(wlist, corpus_vocab, inv_index, corpus_size, max_words_per_topic):
    num_pairs = 0
    pmi = 0.0
    npmi = 0.0
    # compute topic coherence only for first 10 word in each topic.
    wlist = wlist[:max_words_per_topic]
    for (w1, w2) in itertools.combinations(wlist, 2):
        w1docs = get_docs_from_index(w1, corpus_vocab, inv_index)
        w2docs = get_docs_from_index(w2, corpus_vocab, inv_index)
        if len(w1docs) and len(w2docs):
            word_pair_pmi, word_pair_npmi = get_pmi(w1docs, w2docs, corpus_size)
            pmi += word_pair_pmi
            npmi += word_pair_npmi
            num_pairs += 1
    if num_pairs:
        pmi /= num_pairs
        npmi /= num_pairs
    return pmi, npmi, num_pairs


def calc_pmi_for_all_topics(topic_dict, corpus_vocab, inv_index, corpus_size):
    pmi_dict = dict()
    npmi_dict = dict()
    for k in topic_dict.keys():
        wlist = topic_dict[k]
        use_N_words = len(wlist)  # use full list
        pmi, npmi, _ = get_topic_pmi(wlist, corpus_vocab, inv_index, corpus_size, use_N_words)
        # print(npmi, pmi)
        # print(wlist)
        pmi_dict[k] = pmi
        npmi_dict[k] = npmi

    return pmi_dict, npmi_dict


def launch_socket(port=1234, ref_corpus=None):
    print_header('Launching socket at port {}'.format(port))
    # create a socket object
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # get local machine name
    # host = socket.gethostname()
    host = socket.gethostbyname('localhost')

    # bind to the port
    serversocket.bind((host, port))

    # queue up to 5 requests
    serversocket.listen(5)
    print_header('Socket ready port {}'.format(port))

    # set up socket first, then load corpus
    if ref_corpus is None:
        ref_corpus = RefCorpus()
        ref_corpus.load_corpus()

    test_pmi(ref_corpus.inv_index, ref_corpus.corpus_vocab, ref_corpus.corpus_size)
    # test_idf(rc.inv_index, rc.corpus_vocab, rc.corpus_size)
    print_header('Test done')

    while True:
        # establish a connection
        clientsocket, addr = serversocket.accept()
        start = time.time()

        print("Got a connection from %s" % str(addr))
        # currentTime = time.ctime(time.time()) + "\r\n"
        # clientsocket.send(currentTime.encode('ascii'))

        # json_str = clientsocket.recv(10 * 1024 * 1024)
        # topic_json = json.loads(json_str.decode('ascii'))

        # https://stackoverflow.com/questions/24726495/pickle-eoferror-ran-out-of-input-when-recv-from-a-socket
        # data = b"".join(iter(partial(clientsocket.recv, 1024 * 1024), b""))
        # topic_dict = pickle.loads(data)
        # data = []
        # while True:
        #     packet = s.recv(1024 * 1024)
        #     if not packet: break
        #         data.append(packet)
        # topic_dict = pickle.loads(b"".join(data))
        try:
            data = clientsocket.recv(1024 * 1024)
            # data = []
            # while True:
            #     print('waiting for packet # {0}'.format(len(data)))
            #     packet = clientsocket.recv(4096)
            #     print(packet)
            #     wait = len(packet)
            #     if not packet:
            #         break
            #     data.append(packet)
            #     print('received packet # {0}'.format(len(data)))
            # print('got here')
            # print(data)
            # data = b"".join(data)
            # data = clientsocket.recv(4096)
            received = pickle.loads(data)
            print(received)
            if isinstance(received, str):
                filename, i = received.split(':')
                topic_dict = pickle.load(open(filename,'rb'))['Topic Words'][int(i)]
            else:
                topic_dict = received
            # print('received data: {0} Mb'.format(len(data)))
            print('received data')
            print('Time elapsed: {:.2f}s'.format(time.time() - start))
            print(topic_dict)
            pmi_dict, npmi_dict = calc_pmi_for_all_topics(topic_dict,
                                                          ref_corpus.corpus_vocab,
                                                          ref_corpus.inv_index,
                                                          ref_corpus.corpus_size)

            print('completed calculation')
            print('Time elapsed: {:.2f}s'.format(time.time() - start))
            result_dict = {'pmi_dict': pmi_dict, 'npmi_dict': npmi_dict}

            res = pickle.dumps(result_dict)
            clientsocket.send(res, )
            # clientsocket.shutdown(socket.SHUT_RDWR)
            clientsocket.close()
            del clientsocket
            print('Connection closed')
            print('Time elapsed: {:.2f}s'.format(time.time() - start))
        except Exception as e:
            print('Error occured when receiving packet')
            print(e)
            # clientsocket.shutdown(socket.SHUT_RDWR)
            clientsocket.close()
            del clientsocket


def request_pmi(topic_dict, port=1234):
    try:
        # create a socket object
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # get local machine name
        host = socket.gethostname()

        # connection to hostname on the port.
        s.connect((host, port))

        # json_str = json.dumps(topic_json)
        # s.send(json_str.encode('ascii'))

        s.send(pickle.dumps(topic_dict), )

        # Receive no more than 1024 * 1024 bytes
        # res = s.recv(1024*1024*1024)
        data = []
        while True:
            packet = s.recv(4096)
            if not packet:
                break
            data.append(packet)
        res_dict = pickle.loads(b"".join(data))

        s.close()
        pmi_dict = res_dict['pmi_dict']
        npmi_dict = res_dict['npmi_dict']
    except:
        print_error('Failed to run NPMI calc, NPMI and PMI set to 0.0')
        pmi_dict = dict()
        npmi_dict = dict()
        for k in topic_dict:
            pmi_dict[k] = 0
            npmi_dict[k] = 0

    return pmi_dict, npmi_dict


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', dest='port', action='store', type=int, default=1234,
                        help='port number to launch socket')
    args = parser.parse_args()
    print('port is set to', args.port)

    launch_socket(port=args.port)
