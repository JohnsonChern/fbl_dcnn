# -*- coding:<utf-8> -*-

import data_utils
import mxnet as mx
import numpy as np
import copy
import os, sys

class DataIter(mx.io.DataIter):
    def __init__(self, src_file, src_tp_file,
                 tar_file, tar_tp_file, batch_size, data_name):
        super(DataIter, self).__init__()
        self.data_name = data_name
        self.batch_size = batch_size
        src, src_len, src_tp, tar, tar_len, tar_tp, tp_len, vocab\
            = data_utils.mon_lingual_input(src_file, src_tp_file, tar_file, tar_tp_file)
        self.data = [src, tar, src_tp, tar_tp]
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
            bdata = self.data[
                index*self.batch_size : (index + 1)*self.batch_size,
                0 : seq_len]
            # blabel = self.label[
            #     index * self.batch_size : (index + 1)*self.batch_size]
            data_all = [mx.nd.array(bdata)]
            # label_all = [mx.nd.array(blabel)]
            data_names = [self.data_name]
            label_names = [self.label_name]
            data_batch = Batch(data_names, data_all, seq_len)
            yield data_batch

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
