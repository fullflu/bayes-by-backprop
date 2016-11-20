#!/usr/bin/env python
# coding: utf-8

import chainer.functions as F
from chainer import Variable, optimizers
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
import net
import six
import os
import time

def mnist_preprocessing(sample_N = 600000, test_ratio = 0.25):
    if os.path.exists("mnist_preprocessed_data.npy"):
        x = np.load("mnist_preprocessed_data.npy")
        y = np.load("mnist_preprocessed_target.npy")
        #y = np.int32(preprocessing.OneHotEncoder(sparse=False).fit_transform(y.reshape(y.shape[0],1)))
        idx = np.random.choice(x.shape[0], sample_N)
        x = x[idx]
        y = y[idx]
    else:
        mnist = fetch_mldata('MNIST original')
        x = np.float32(mnist.data[:]) / 126.
        np.save("mnist_preprocessed_data",x)
        y = np.int32(mnist.target)
        #y = np.int32(preprocessing.OneHotEncoder(sparse=False).fit_transform(y.reshape(y.shape[0],1)))
        np.save("mnist_preprocessed_target",y)
        idx = np.random.choice(x.shape[0], sample_N)
        x = x[idx]
        y = y[idx]

    tr_idx, te_idx = train_test_split(np.arange(sample_N), test_size = test_ratio)
    tr_x, te_x = x[tr_idx], x[te_idx]
    tr_y, te_y = y[tr_idx], y[te_idx]

    return tr_x,te_x,tr_y,te_y

def get_gaussianloglikelihood_pw(x,mu,sigma):
    return -0.5 * np.log(2*np.pi) - np.log(sigma) - (x - mu)**2 / (2 * sigma**2)

def get_gaussianloglikelihood_qw(x,mu,sigma):
    return -0.5 * np.log(2*np.pi) - F.log(sigma) - (x - mu)**2 / (2 * sigma**2)

"""
sample_N = 6000
test_ratio = 0.25 
tr_x, te_x, tr_y, te_y = mnist_preprocessing(sample_N, test_ratio)
"""

class BBP_agent(object):
    """docstring for BPP_agent"""
    def __init__(self, model_num = 3, sample_N = 60000, test_ratio = .25, batch_size = 32, max_iter = 100):
        super(BBP_agent, self).__init__()
        self.sample_N = sample_N
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.model_num = model_num

    def prepare_data(self):            
        self.tr_x, self.te_x, self.tr_y, self.te_y = mnist_preprocessing(self.sample_N, self.test_ratio)    
        self.tr_N = self.tr_x.shape[0]
        self.M = self.tr_N / float(self.batch_size)

    def set_model_parameter(self):
        self.prior_ratio = np.float32(0.5)
        self.prior_sigma_1 = np.float32(np.exp(-1))
        self.prior_sigma_2 = np.float32(np.exp(-7))
        self.n_in = self.tr_x.shape[1]
        self.n_hidden1 = 200
        self.n_hidden2 = 200
        self.n_out = 10
        self.prior_pho_var = np.float32(.05)
        self.model = net.MLP_MNIST_bbp(self.n_in, self.n_hidden1, self.n_hidden2, self.n_out, self.prior_ratio, 
            self.prior_sigma_1, self.prior_sigma_2, self.prior_pho_var)


    def pD_w(self,Data_indices):
        in_data = Variable(self.tr_x[Data_indices])
        t = Variable(self.tr_y[Data_indices])
        cross_entropy = 0.
        for i in range(self.model_num):
            w1,w2,w3 = self.models[i]
            w1 = F.reshape(w1,(self.n_in,self.n_hidden1))
            w2 = F.reshape(w2,(self.n_hidden1,self.n_hidden2))
            w3 = F.reshape(w3,(self.n_hidden2,self.n_out))
            h1 = F.relu(F.matmul(in_data,w1))
            h2 = F.relu(F.matmul(h1,w2))
            pred = F.softmax(F.matmul(h2,w3))
            cross_entropy += F.softmax_cross_entropy(pred,t)
        return -1 * cross_entropy

    def KL_minibatch(self):
        log_qw_theta_sum = 0.
        log_pw_sum = 0.
        for i in range(self.model_num):
            w1,w2,w3 = self.models[i]
            w = F.hstack([w1,w2,w3])
            log_qw_theta = get_gaussianloglikelihood_qw(w,self.model.mu_hstack(),self.model.sigma_hstack())
            #log_qw_theta_sum += F.sum(log_qw_theta, axis=1)
            log_qw_theta_sum += F.sum(log_qw_theta)
            log_pw = self.prior_ratio * get_gaussianloglikelihood_pw(w,0,self.prior_sigma_1) + (1 - self.prior_ratio) * get_gaussianloglikelihood_pw(w,0,self.prior_sigma_2)
            #log_pw_sum += F.sum(log_pw, axis=1)
            log_pw_sum += F.sum(log_pw)
        return (log_qw_theta_sum - log_pw_sum) / self.M


    def fit(self):
        now = time.time()
        for iter_ in range(self.max_iter):
            perm_tr = np.random.permutation(self.tr_N)
            for batch_idx in six.moves.range(0,self.tr_N,self.batch_size):
                print("start_batch:{}".format(batch_idx))
                Data_indices = perm_tr[batch_idx:batch_idx + self.batch_size]
                #self.model.zerograds()
                self.models = []
                for i in range(self.model_num):
                    self.models.append(self.model())
                print("finish_models_append")
                start_f_calc = time.time()
                t1 = self.KL_minibatch()
                t2 = self.pD_w(Data_indices)
                f_batch = t1 - t2
                end_f_calc = time.time()
                print("finish_f_calculation:{}".format(end_f_calc -start_f_calc))
                print("f_batch_grad:{}".format(f_batch.grad))
                #print("mu1_grad:{}".format(self.model.mu1.grad.shape))                
                f_batch.backward()
                print("f_batch_grad:{}".format(f_batch.grad))
                print("finish_f_backward:{}".format(time.time() - end_f_calc))
                print("mu1_grad:{}".format(self.model.mu1.grad.shape))                
                self.model.update()
            print(iter_,f_batch.data)

class BBP_agent2(object):
    """docstring for BPP_agent"""
    def __init__(self, model_num = 3, sample_N = 60000, test_ratio = .25, batch_size = 32, max_iter = 100):
        super(BBP_agent2, self).__init__()
        self.sample_N = sample_N
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.model_num = model_num

    def prepare_data(self):            
        self.tr_x, self.te_x, self.tr_y, self.te_y = mnist_preprocessing(self.sample_N, self.test_ratio)    
        self.tr_N = self.tr_x.shape[0]
        self.M = self.tr_N / float(self.batch_size)

    def set_model_parameter(self):
        self.prior_ratio = np.float32(0.5)
        self.prior_sigma_1 = np.float32(np.exp(-1))
        self.prior_sigma_2 = np.float32(np.exp(-7))
        self.n_in = self.tr_x.shape[1]
        self.n_hidden1 = 500
        self.n_hidden2 = 500
        self.n_out = 10
        self.prior_pho_var = np.float32(.05)
        self.model = net.MLP_MNIST_bbp(self.n_in, self.n_hidden1, self.n_hidden2, self.n_out, self.prior_ratio, 
            self.prior_sigma_1, self.prior_sigma_2, self.prior_pho_var)
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)


    def pD_w(self,Data_indices):
        in_data = Variable(self.tr_x[Data_indices])
        t = Variable(self.tr_y[Data_indices])
        cross_entropy = 0.
        for i in range(self.model_num):
            w1,w2,w3 = self.models[i]
            w1 = F.reshape(w1,(self.n_in,self.n_hidden1))
            w2 = F.reshape(w2,(self.n_hidden1,self.n_hidden2))
            w3 = F.reshape(w3,(self.n_hidden2,self.n_out))
            h1 = F.relu(F.matmul(in_data,w1))
            h2 = F.relu(F.matmul(h1,w2))
            pred = F.softmax(F.matmul(h2,w3))
            cross_entropy += F.softmax_cross_entropy(pred,t)
        return -1 * cross_entropy

    def KL_minibatch(self):
        log_qw_theta_sum = 0.
        log_pw_sum = 0.
        w = self.model.w_hstack()
        log_qw_theta = get_gaussianloglikelihood_qw(w,self.model.mu_hstack(),self.model.sigma_hstack())
        #log_qw_theta_sum += F.sum(log_qw_theta, axis=1)
        log_qw_theta_sum += F.sum(log_qw_theta)
        log_pw = self.prior_ratio * get_gaussianloglikelihood_pw(w,0,self.prior_sigma_1) + (1 - self.prior_ratio) * get_gaussianloglikelihood_pw(w,0,self.prior_sigma_2)
        #log_pw_sum += F.sum(log_pw, axis=1)
        log_pw_sum += F.sum(log_pw)
        return (log_qw_theta_sum - log_pw_sum) / (self.M * self.model_num)

    def fit(self):
        now = time.time()
        for iter_ in range(self.max_iter):
            perm_tr = np.random.permutation(self.tr_N)
            for batch_idx in six.moves.range(0,self.tr_N,self.batch_size):
                #print("start_batch:{}".format(batch_idx))
                Data_indices = perm_tr[batch_idx:batch_idx + self.batch_size]
                #self.model.zerograds()
                self.models = []
                f_batch_mean = 0.
                for i in range(self.model_num):
                    self.model.zerograds()
                    #self.models.append(self.model())
                #print("finish_models_append")
                    start_f_calc = time.time()
                    #t2 = self.pD_w(Data_indices)
                    in_data = Variable(self.tr_x[Data_indices])
                    t = Variable(self.tr_y[Data_indices])
                    #print(t.data)
                    t2 = self.model(in_data,t)
                    t1 = self.KL_minibatch()
                
                    f_batch = t1 - t2
                    print("t1:{}".format(t1.data))
                    print("t2:{}".format(t2.data))
                    end_f_calc = time.time()
                    #print("finish_f_calculation:{}".format(end_f_calc -start_f_calc))
                    #print("f_batch_grad:{}".format(f_batch.grad))
                #print("mu1_grad:{}".format(self.model.mu1.grad.shape))                
                    f_batch.backward(retain_grad = True)
                    #print("f_batch_grad:{}".format(f_batch.grad))
                    #print("finish_f_backward:{}".format(time.time() - end_f_calc))
                    #print("mu1_grad:{}".format(self.model.mu1.W.grad.shape))                
                    #self.model.update(self.model_num)
                    self.optimizer.update()
                    f_batch_mean += f_batch.data
                print("f_batch_mean:{}".format(f_batch_mean/float(self.model_num)))
            print(iter_,f_batch.data)

agent = BBP_agent2(sample_N =  6000)
agent.prepare_data()
print("finish data preparation!!")
agent.set_model_parameter()
agent.fit()




"""
    u1 = prior_ratio * np.random.normal(0,prior_sigma_1**2,n_in * n_hidden1) + (1 - prior_ratio) * np.random.normal(0,prior_sigma_1**2,n_in * n_hidden1)
    u1 = u1.reshape((n_in, n_hidden1)).astype(np.float32)
    u2 = prior_ratio * np.random.normal(0,prior_sigma_1**2,n_hidden1 * n_hidden2) + (1 - prior_ratio) * np.random.normal(0,prior_sigma_1**2,n_hidden1 * n_hidden2)
    u2 = u1.reshape((n_hidden1, n_hidden2)).astype(np.float32)
    u3 = prior_ratio * np.random.normal(0,prior_sigma_1**2,n_hidden2 * 10) + (1 - prior_ratio) * np.random.normal(0,prior_sigma_1**2,n_hidden2 * 10)
    u3 = u1.reshape((n_hidden2, n_out)).astype(np.float32)
"""








