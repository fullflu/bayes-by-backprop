#!/usr/bin/env python
# coding: utf-8

from chainer import Chain
import chainer.functions as F
import chainer.links as L
from chainer import Variable
import numpy as np

class MLP_MNIST_bbp(Chain):
    def __init__(self, n_in = 784, n_hidden1 = 1200, n_hidden2 = 1200, n_out = 10, lr = 1e-4, prior_ratio = 0.5, prior_sigma_1 = np.exp(-1), prior_sigma_2 = np.exp(-7), prior_pho_var = .05):
        super(MLP_MNIST_bbp, self).__init__(
            w1 = L.Linear(n_in, n_hidden1),
            w2 = L.Linear(n_hidden1, n_hidden2),
            w3 = L.Linear(n_hidden2, n_out),
            mu1 = L.Linear(n_in, n_hidden1),
            mu2 = L.Linear(n_hidden1, n_hidden2),
            mu3 = L.Linear(n_hidden2, n_out),
            pho1 = L.Linear(n_in, n_hidden1),
            pho2 = L.Linear(n_hidden1, n_hidden2),
            pho3 = L.Linear(n_hidden2, n_out),
            )
        self.n_in = n_in
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_out = n_out
        self.lr = lr

        #mu1 = prior_ratio * np.random.normal(0,prior_sigma_1,n_in * n_hidden1) + (1 - prior_ratio) * np.random.normal(0,prior_sigma_1,n_in * n_hidden1)
        #mu1 = mu1.reshape((n_in, n_hidden1)).astype(np.float32)
        #mu2 = prior_ratio * np.random.normal(0,prior_sigma_1,n_hidden1 * n_hidden2) + (1 - prior_ratio) * np.random.normal(0,prior_sigma_1,n_hidden1 * n_hidden2)
        #mu2 = mu2.reshape((n_hidden1, n_hidden2)).astype(np.float32),
        #mu3 = prior_ratio * np.random.normal(0,prior_sigma_1,n_hidden2 * n_out) + (1 - prior_ratio) * np.random.normal(0,prior_sigma_1,n_hidden2 * n_out)
        #mu3 = mu3.reshape((n_hidden2, 10)).astype(np.float32)

        for i,m in enumerate([self.mu1,self.mu2,self.mu3]):
            tmp_w = prior_ratio * np.random.normal(0,prior_sigma_1,m.W.shape[1] * m.W.shape[0]) + (1 - prior_ratio) * np.random.normal(0,prior_sigma_2,m.W.shape[1] * m.W.shape[0])
            m.W = Variable(tmp_w.reshape(m.W.shape).astype(np.float32))
            tmp_b = prior_ratio * np.random.normal(0,prior_sigma_1,m.b.shape[0]) + (1 - prior_ratio) * np.random.normal(0,prior_sigma_2,m.b.shape[0])
            m.b = Variable(tmp_b.reshape(m.b.shape).astype(np.float32))

        for i,m in enumerate([self.pho1,self.pho2,self.pho3]):
            tmp_w = np.random.normal(0,prior_pho_var,m.W.shape[1] * m.W.shape[0])
            m.W = Variable(tmp_w.reshape(m.W.shape).astype(np.float32))
            tmp_b = np.random.normal(0,prior_pho_var,m.b.shape[0])
            m.b = Variable(tmp_b.reshape(m.b.shape).astype(np.float32))

        #for i,w in enumerate([self.w1,self.w2,self.w3]):

        """
        self.mu1 = Variable(mu1.astype(np.float32))
        self.mu2 = Variable(mu2.astype(np.float32))
        self.mu3 = Variable(mu3.astype(np.float32))
        
        pho1 = np.random.normal(0,prior_pho_var,n_in * n_hidden1)
        #pho1 = Variable(pho1.reshape((n_in, n_hidden1)).astype(np.float32))
        self.pho1 = Variable(pho1.astype(np.float32))
        pho2 = np.random.normal(0,prior_pho_var,n_hidden1 * n_hidden2)
        #pho2 = Variable(pho2.reshape((n_hidden1, n_hidden2)).astype(np.float32))
        self.pho2 = Variable(pho2.astype(np.float32))
        pho3 = np.random.normal(0,prior_pho_var,n_hidden2 * n_out)
        #pho3 = Variable(pho3.reshape((n_hidden2, 10)).astype(np.float32))
        self.pho3 = Variable(pho3.astype(np.float32))
        #L.add_parames("pho1",)
        """

    def __call__(self,x,t):
        
        self.eps1_w = np.random.normal(0,1,self.mu1.W.shape).astype(np.float32)
        self.eps2_w = np.random.normal(0,1,self.mu2.W.shape).astype(np.float32)
        self.eps3_w = np.random.normal(0,1,self.mu3.W.shape).astype(np.float32)
        #self.eps1_w = eps1_w.reshape(self.m1.W.shape)
        #self.eps2_w = eps2_w.reshape(self.m2.W.shape)
        #self.eps3_w = eps3_w.reshape(self.m3.W.shape)

        self.w1.W = self.mu1.W + F.log(1 + F.exp(self.pho1.W))*Variable(self.eps1_w)
        self.w2.W = self.mu2.W + F.log(1 + F.exp(self.pho2.W))*Variable(self.eps2_w)
        self.w3.W = self.mu3.W + F.log(1 + F.exp(self.pho3.W))*Variable(self.eps3_w)

        self.eps1_b = np.random.normal(0,1,self.mu1.b.shape).astype(np.float32)
        self.eps2_b = np.random.normal(0,1,self.mu2.b.shape).astype(np.float32)
        self.eps3_b = np.random.normal(0,1,self.mu3.b.shape).astype(np.float32)

        self.w1.b = self.mu1.b + F.log(1 + F.exp(self.pho1.b))*Variable(self.eps1_b)
        self.w2.b = self.mu2.b + F.log(1 + F.exp(self.pho2.b))*Variable(self.eps2_b)
        self.w3.b = self.mu3.b + F.log(1 + F.exp(self.pho3.b))*Variable(self.eps3_b)

        h1 = F.relu(self.w1(x))
        h2 = F.relu(self.w2(h1))
        h3 = self.w3(h2)

        self.eps1 = [self.eps1_w,self.eps1_b]
        self.eps2 = [self.eps2_w,self.eps2_b]
        self.eps3 = [self.eps3_w,self.eps3_b]

        #print("w1_shape:{}".format(w1.shape))
        """
        w1 = F.reshape(w1,(self.n_in,self.n_hidden1))
        w2 = F.reshape(w2,(self.n_hidden1,self.n_hidden2))
        w3 = F.reshape(w3,(self.n_hidden2,self.n_out))
        """
        #return w1,w2,w3
        #print h3.shape,t.shape
        #h3 = F.reshape(h3,(h3.shape[0],))
        #print h3.shape,t.shape
        return F.softmax_cross_entropy(h3,t)

    def mu_hstack(self):
        return F.hstack([F.flatten(self.mu1.W),F.flatten(self.mu1.b),F.flatten(self.mu2.W),F.flatten(self.mu2.b),F.flatten(self.mu3.W),F.flatten(self.mu3.b)])

    def w_hstack(self):
        return F.hstack([F.flatten(self.w1.W),F.flatten(self.w1.b),F.flatten(self.w2.W),F.flatten(self.w2.b),F.flatten(self.w3.W),F.flatten(self.w3.b)])

    def sigma_hstack(self):
        return F.log(1 + F.exp(F.hstack([F.flatten(self.pho1.W),F.flatten(self.pho1.b),F.flatten(self.pho2.W),F.flatten(self.pho2.b),F.flatten(self.pho3.W),F.flatten(self.pho3.b)])))

    def update(self,model_num):
        """
        print("update:{}".format(self.mu1.W.grad.shape))
        print("update:{}".format(self.mu2.W.grad.shape))
        print("update:{}".format(self.mu3.W.grad.shape))
        print("update:{}".format(self.pho1.W.grad.shape))
        """
        for m,w in zip([self.mu1,self.mu2,self.mu3],[self.w1,self.w2,self.w3]):
            delta_w = m.W.grad + w.W.grad
            m.W = m.W - self.lr * delta_w
            delta_b = m.b.grad + w.b.grad
            m.b = m.b - self.lr * delta_b

        for pho,w,eps in zip([self.pho1,self.pho2,self.pho3],[self.w1,self.w2,self.w3],[self.eps1,self.eps2,self.eps3]):
            delta_w = pho.W.grad + w.W.grad * eps[0] / (1 + F.exp(-1*pho.W))
            pho.W = pho.W - self.lr * delta_w / np.float32(model_num)
            delta_b = pho.b.grad + w.b.grad * eps[1] / (1 + F.exp(-1*pho.b))
            pho.b = pho.b - self.lr * delta_b / np.float32(model_num)
        #print("update:{}".format(self.mu1.grad.shape))
        #print("mu1_shape:{}".format(self.mu1.shape))

class MLP_MNIST_bbp_(object):
    def __init__(self, n_in = 784, n_hidden1 = 1200, n_hidden2 = 1200, n_out = 10, lr = 1e-4, prior_ratio = 0.5, prior_sigma_1 = np.exp(-1), prior_sigma_2 = np.exp(-7), prior_pho_var = .05):
        super(MLP_MNIST_bbp, self).__init__()
        self.n_in = n_in
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_out = n_out
        self.lr = lr

        mu1 = prior_ratio * np.random.normal(0,prior_sigma_1,n_in * n_hidden1) + (1 - prior_ratio) * np.random.normal(0,prior_sigma_1,n_in * n_hidden1)
        #mu1 = mu1.reshape((n_in, n_hidden1)).astype(np.float32)
        mu2 = prior_ratio * np.random.normal(0,prior_sigma_1,n_hidden1 * n_hidden2) + (1 - prior_ratio) * np.random.normal(0,prior_sigma_1,n_hidden1 * n_hidden2)
        #mu2 = mu2.reshape((n_hidden1, n_hidden2)).astype(np.float32),
        mu3 = prior_ratio * np.random.normal(0,prior_sigma_1,n_hidden2 * n_out) + (1 - prior_ratio) * np.random.normal(0,prior_sigma_1,n_hidden2 * n_out)
        #mu3 = mu3.reshape((n_hidden2, 10)).astype(np.float32)
        self.mu1 = Variable(mu1.astype(np.float32))
        self.mu2 = Variable(mu2.astype(np.float32))
        self.mu3 = Variable(mu3.astype(np.float32))
        pho1 = np.random.normal(0,prior_pho_var,n_in * n_hidden1)
        #pho1 = Variable(pho1.reshape((n_in, n_hidden1)).astype(np.float32))
        self.pho1 = Variable(pho1.astype(np.float32))
        pho2 = np.random.normal(0,prior_pho_var,n_hidden1 * n_hidden2)
        #pho2 = Variable(pho2.reshape((n_hidden1, n_hidden2)).astype(np.float32))
        self.pho2 = Variable(pho2.astype(np.float32))
        pho3 = np.random.normal(0,prior_pho_var,n_hidden2 * n_out)
        #pho3 = Variable(pho3.reshape((n_hidden2, 10)).astype(np.float32))
        self.pho3 = Variable(pho3.astype(np.float32))
        """
        mu1 = L.Linear(n_in, n_hidden1),
        mu2 = L.Linear(n_hidden1, n_hidden2),
        mu3 = L.Linear(n_hidden2, 10),
        """
        #bnorm1 = L.BatchNormalization(n_hidden1),
        #bnorm2 = L.BatchNormalization(n_hidden2)

    def __call__(self):
        eps1 = np.random.normal(0,1,self.n_in*self.n_hidden1).astype(np.float32)
        eps2 = np.random.normal(0,1,self.n_hidden1*self.n_hidden2).astype(np.float32)
        eps3 = np.random.normal(0,1,self.n_hidden2*self.n_out).astype(np.float32)

        w1 = self.mu1 + F.log(1 + F.exp(self.pho1))*Variable(eps1)
        w2 = self.mu2 + F.log(1 + F.exp(self.pho2))*Variable(eps2)
        w3 = self.mu3 + F.log(1 + F.exp(self.pho3))*Variable(eps3)
        #print("w1_shape:{}".format(w1.shape))
        """
        w1 = F.reshape(w1,(self.n_in,self.n_hidden1))
        w2 = F.reshape(w2,(self.n_hidden1,self.n_hidden2))
        w3 = F.reshape(w3,(self.n_hidden2,self.n_out))
        """
        return w1,w2,w3

    def mu_hstack(self):
        return F.hstack([self.mu1,self.mu2,self.mu3])

    def sigma_hstack(self):
        return F.log(1 + F.exp(F.hstack([self.pho1,self.pho2,self.pho3])))

    def update(self):
        print("update:{}".format(self.mu1.grad.shape))
        print("update:{}".format(self.mu2.grad.shape))
        print("update:{}".format(self.mu3.grad.shape))
        print("update:{}".format(self.pho1.grad.shape))
        self.mu1 = self.mu1 - self.lr * self.mu1.grad
        self.mu2 = self.mu2 - self.lr * self.mu2.grad
        self.mu3 = self.mu3 - self.lr * self.mu3.grad
        self.pho1 = self.pho1 - self.lr * self.pho1.grad
        self.pho2 = self.pho2 - self.lr * self.pho2.grad
        self.pho3 = self.pho3 - self.lr * self.pho3.grad
        #print("update:{}".format(self.mu1.grad.shape))
        #print("mu1_shape:{}".format(self.mu1.shape))

class MLP_MNIST_dropput(Chain):
    def __init__(self, n_in = 784, n_hidden1 = 1200, n_hidden2 = 1200):
        super(MLPListNet, self).__init__(
            l1 = L.Linear(n_in, n_hidden1),
            l2 = L.Linear(n_hidden1, n_hidden2),
            l3 = L.Linear(n_hidden2, 10),
            #bnorm1 = L.BatchNormalization(n_hidden1),
            #bnorm2 = L.BatchNormalization(n_hidden2)
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h1 = F.dropout(h1)
        h2 = F.relu(self.l2(h1))
        h2 = F.dropout(h2)
        #h1 = F.relu(self.bnorm1(self.l1(x)))
        #h1 = F.dropout(h1)
        #h2 = F.relu(self.bnorm2(self.l2(h1)))
        #h2 = F.dropout(h2)
        return self.l3(h2)

