from __future__ import print_function
import numpy as np
import os

dataset_path = 'output/preprocessed_blink_features/'

X_train = np.load(dataset_path + 'Blinks_w30_s2.npy')
y_train = np.load(dataset_path + 'Labels_w30_s2.npy')

X_val = np.load(dataset_path + 'BlinksVal_w30_s2.npy')
y_val = np.load(dataset_path + 'LabelsVal_w30_s2.npy')

X_test = np.load(dataset_path + 'BlinksTest_w30_s2.npy')
y_test = np.load(dataset_path + 'LabelsTest_w30_s2.npy')

X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')

def do_z_score_normalization(X_train, X_val, feat_num):
    train_feature = ((X_train[:,:,feat_num] - np.mean(X_train[:,:,feat_num])) 
        / np.std(X_train[:,:,feat_num]))
    val_feature = ((X_val[:,:,feat_num] - np.mean(X_train[:,:,feat_num])) 
        / np.std(X_train[:,:,feat_num]))
    test_feature = ((X_test[:,:,feat_num] - np.mean(X_train[:,:,feat_num])) 
        / np.std(X_train[:,:,feat_num]))
    
    return train_feature, val_feature, test_feature

def z_score_normalization(X_train, X_val, X_test):
    features_count = X_train.shape[2]
    for i in range(features_count):
        X_train[:,:,i], X_val[:,:,i], X_test[:,:,i] = (
            do_z_score_normalization(X_train, X_val, i))

z_score_normalization(X_train, X_val, X_test)

train_count = len(X_train)

shuffled_dataset_index_list = np.arange(train_count)
np.random.shuffle(shuffled_dataset_index_list) 

class BatchGenerator():
    def __init__(self, batch_size_):
        self.result = []
        self.batch_size_ = batch_size_
        self.index = 0

    def generate(self): 
        batch_data_  = np.zeros((self.batch_size_, X_train.shape[1], X_train.shape[2]), dtype = np.float32)
        batch_label_ = np.zeros((self.batch_size_), dtype=np.int32)

        for i in range(self.batch_size_):
            batch_data_[i] = X_train[shuffled_dataset_index_list[self.index]]
            batch_label_[i] = y_train[shuffled_dataset_index_list[self.index]]

            self.index += 1
            if self.index == train_count:
                self.index = 0
                np.random.shuffle(shuffled_dataset_index_list)
            
        batch_data_ = np.asarray(batch_data_, dtype = np.float32)
        
        return batch_data_, batch_label_ 

class DataHandler(object):

    def __init__(self, batch_size):
        self.batch_size_ = batch_size        
        self.data_generator = BatchGenerator(batch_size)


    def get_batch(self):
        return self.data_generator.generate()

    def get_dataset_size(self):
        return train_count


class EvalBatchGenerator():

    def __init__(self, batch_size_):
        self.result = []
        self.batch_size_ = batch_size_
        self.index = 0
        self.indices = np.arange(len(y_val))
        np.random.shuffle(self.indices)
    
    def generate(self):    
        batch_data_  = np.zeros((self.batch_size_, X_val.shape[1], X_val.shape[2]), dtype = np.float32)
        batch_label_ = np.zeros((self.batch_size_), dtype = np.int32)

        if self.index + self.batch_size_ > len(y_val):
            batch_data_[:len(y_val) - self.index] = X_val[self.indices[self.index : len(y_val)]]
            batch_label_[:len(y_val) - self.index] = y_val[self.indices[self.index : len(y_val)]]
            needed = self.batch_size_ - (len(y_val) - self.index)
            batch_data_[len(y_val) - self.index:] = X_val[self.indices[:needed]]
            batch_label_[len(y_val) - self.index:] = y_val[self.indices[:needed]]
            self.index = needed
        else:
            batch_data_ = X_val[self.indices[self.index : self.index + self.batch_size_]]
            batch_label_ = y_val[self.indices[self.index : self.index + self.batch_size_]]
            self.index += self.batch_size_
        
        if self.index == len(y_val):
            self.index=0
        
        return batch_data_, batch_label_ 


class EvalDataHandler(object):

    def __init__(self, batch_size):
        self.batch_size_ = batch_size        
        self.data_generator = EvalBatchGenerator(batch_size)


    def get_batch(self):
        return self.data_generator.generate()

    def get_dataset_size(self):
        return len(y_val)

class TestDataHandler(object):
    def get_all(self):
        return X_test, y_test

    def get_dataset_size(self):
        return len(y_test)