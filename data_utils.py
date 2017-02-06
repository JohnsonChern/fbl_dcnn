# -*- coding:<utf-8> -*-

import xlrd
import numpy as np
from collections import defaultdict
from bidict import namedbidict

Vocab = namedbidict('Vocab', 'word', 'id')
SRC_UNLABEL = "./data/src_unlabel.xlsx"
TAR_UNLABEL = "./data/tar_unlabel.xlsx"
SRC_TP = "./data/src_tp.csv"
TAR_TP = "./data/tar_tp.csv"

def read_excel(in_file, labeled=False):
    """
    read a xlsx file
    """
    data = xlrd.open_workbook(in_file).sheets()[0]
    docs = []
    labels = []
    for i in range(data.nrows):
        row_values = data.row_values(i)
        doc = row_values[0].split()
        docs.append(doc)
        if labeled:
            labels.append(row_values[1])
    if labeled:
        labels = np.asarray(labels, dtype=np.int32)
        return docs, labels
    else:
        return docs

def read_tp(in_file):
    """
    read a tp.csv file
    return:
        tp_data: list of lists
        vocab: construct a vocabulary using the first column of tp.csv file
    """
    vocab = Vocab()
    tp_data = {}
    file = open(in_file, 'r')
    file.readline()
    index = 1
    for line in file.read().splitlines():
        splt = line.split(",")
        word = splt[0][1:-1]
        tp_data[word] = np.asarray(list(map(float, splt[1:])))
        vocab[word] = index
        index += 1
    return tp_data, vocab

def merge_vocab(voc1, voc2):
    pass

def get_vocab(docs):
    pass

def preprocess_data(data, label=None, padding_word='0'):
    """
    sort docs with their lengths and pad each sentence with '0'
    """
    if label is None:
        label = [-1] * len(data)
    lengths = []
    sorted_index = defaultdict(list)
    max_len = 0

    for index in range(len(data)):
        doc_len = len(data[index])
        sorted_index[doc_len].append(index)
        if doc_len > max_len:
            max_len = doc_len
    
    processed_data = []
    labels = []
    for doc_len in sorted(sorted_index.keys()):
        indexes = sorted_index[doc_len]
        pad_num = max_len - doc_len
        for index in indexes:
            processed_data.append(data[index] + [padding_word] * pad_num)
            labels.append(label[index])
            lengths.append(doc_len)

    processed_data = np.asarray(processed_data, dtype=np.int32)
    return processed_data, labels, lengths

def mon_lingual_input():
    src_unlabel = read_excel(SRC_UNLABEL)
    tar_unlabel = read_excel(TAR_UNLABEL)
    src_tp, _ = read_tp(SRC_TP)
    tar_tp, vocab = read_tp(TAR_TP)
    src_unlabel, _, src_len = preprocess_data(src_unlabel)
    tar_unlabel, _, tar_len = preprocess_data(tar_unlabel)

def mul_lingual_input():
    pass
