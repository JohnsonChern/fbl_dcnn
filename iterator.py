# -*- coding:<utf-8> -*-

import data_utils
import mxnet as mx
import numpy as np
import copy
import os, sys

class DataIter(mx.io.DataIter):
    def __init__(self, src_file, src_tp_file,
                 tar_file, tar_tp_file, batch_size, data_name):
        """
        default data_name should be ["src","tar","src_tp","tar_tp"]
        """
        super(DataIter, self).__init__()
        self.data_name = data_name
        self.batch_size = batch_size
        self.src, self.src_len, self.src_tp,\
        self.tar, self.tar_len, self.tar_tp,\
        self.tp_len, self.vocab =\
            data_utils.mon_lingual_input(src_file, src_tp_file, tar_file, tar_tp_file)
        # self.data = [src, tar, src_tp, tar_tp]
        self.lengths = [src_len, tar_len]
        self.n_batches = int(len(self.lengths[0]) / self.batch_size)
        self.permutation = np.random.permutation(self.n_batches)
        self.default_bucket_key = max(self.lengths[0][-1], self.lengths[1][-1])
        self.provide_data = [
            (self.data_name[0], (self.batch_size, self.default_bucket_key)),
            (self.data_name[1], (self.batch_size, self.default_bucket_key)),
            (self.data_name[2], (self.batch_size, self.default_bucket_key, tp_len)),
            (self.data_name[3], (self.batch_size, self.default_bucket_key, tp_len))
        ]

    def __iter__(self):
        for index in self.permutation:
            seq_len = max(self.lengths[0][(index + 1) * self.batch_size - 1],
                          self.lengths[1][(index + 1) * self.batch_size - 1])
            bsrc = self.src[index * self.batch_size: (index + 1) * self.batch_size, 0 : seq_len]
            btar = self.tar[index * self.batch_size: (index + 1) * self.batch_size, 0 : seq_len]
            bsrc, btar = pad(bsrc, btar)
            bsrc_tp = get_batched_tp(bsrc, self.src_tp, self.vocab)
            btar_tp = get_batched_tp(btar, self.tar_tp, self.vocab)
            data_all = list(map(mx.nd.array, [bsrc, btar, bsrc_tp, btar_tp]))
            data_names = [self.data_name]
            data_batch = Batch(data_names, data_all, seq_len)
            yield data_batch

    def pad(mat1, mat2):
            if mat1.shape[1] > mat2.shape[1]:
                pad_col_num = mat1.shape[1] - mat2.shape[1]
                temp = np.zeros((mat2.shape[0], pad_col_num))
                mat2 = np.concatenate((mat2,temp), axis=1)
            elif mat1.shape[1] < mat2.shape[1]:
                pad_col_num = mat2.shape1[1] - mat1.shape[1]
                temp = np.zeros((mat2.shape[0], pad_col_num))
                mat1 = np.concatenate((mat1,temp), axis=1)
            return mat1, mat2

    def get_batched_tp(self, bdocs, tp, vocab):
        """
        given batched docs, corresponding tp and vocabulary, return batched tp data
        parameters type:
            bdocs: batched docs
            tp: corresponding tp, which should be a dict, whose keys are words in string form
                and values are ndarrays of numpy
            vocab: a namebidict that has names 'word' and 'id'
        return:
            btp: batched tp data, which should be list of lists of ndarrays
        """
        btp = []
        for doc in bdocs:
            btp.append(
                list(
                    map(lambda x: tp[x], doc)
                )
            )
        return btp


class Batch(object):
    def __init__(self, data_names, data, bucket_key):
        self.data = data
        # self.label = label
        self.data_names = data_names
        # self.label_names = label_names
        self.bucket_key = bucket_key
    
    @property
    def provide_data(self):
        return []
