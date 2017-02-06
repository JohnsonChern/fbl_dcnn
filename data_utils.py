# -*- coding:<utf-8> -*-

import xlrd
import numpy as np
from bidict import namedbidict

Vocab = namedbidict('Vocab', 'word', 'id')
SRC_UNLABEL = "./data/src_unlabel.xlsx"
TAR_UNLABEL = "./data/tar_unlabel.xlsx"

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

def build_input():
    src_unlabel = read_excel(SRC_UNLABEL)
    tar_unlabel = read_excel(TAR_UNLABEL)
    src_vocab = getVocab(src_unlabel)
    tar_vocab = getVocab(tar_unlabel)
    vocab = merge_vocab(src_vocab, tar_vocab)

