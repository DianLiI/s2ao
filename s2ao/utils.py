import random
import numpy as np
import os, time

SEED = 123
# utils functions
def shuffle(lol, seed=None):
    shuffled_idx = list(range(lol))
    if seed is None:
        random.seed(time.time()%1000)
    else:
        random.seed(seed)
    random.shuffle(shuffled_idx)
    return shuffled_idx

# utils functions
def shuffle2(lengthoflist, max_lengthindex):
    shuffled_idx = [i % max_lengthindex for i in range(lengthoflist)]
    random.seed(time.time() % 1000)
    random.shuffle(shuffled_idx)
    return shuffled_idx

def apply_to_zeros(lst, inner_max_len):
    result = np.zeros([len(lst), inner_max_len], np.float32)
    for i, row in enumerate(lst):
        for j, val in enumerate(row):
            result[i][j] = val
    return result

def relu(x):
    return theano.tensor.switch(x<0, 0, x)

def stable_softmax(yin):
    e_yin = np.exp(yin - yin.max(axis=1, keepdims=True))
    return e_yin / e_yin.sum(axis=1, keepdims=True)

def itemlist(tparams):
    return [vv for kk, vv in tparams.items()]

def numpy_floatX(data):
    return np.asarray(data, dtype=theano.config.floatX)

default_weight_scalar=0.1
# define a lstm rnn
def generate_weight(dim1, dim2, weight_name, weight_scalar=default_weight_scalar):
    if dim1 == 1:
        return theano.shared(name=weight_name,
                         value=weight_scalar * np.random.uniform(-1.0, 1.0, (dim2,))
                         .astype(theano.config.floatX))
    else:
        return theano.shared(name=weight_name,
                         value=weight_scalar * np.random.uniform(-1.0, 1.0,(dim1, dim2))
                         .astype(theano.config.floatX))

def dropout(state_before, use_noise, trng):
    result = T.switch(use_noise, (state_before * trng.binomial(state_before.shape, p=0.5, n=1,
            dtype=state_before.dtype)), state_before * 0.5)
    return result

# helper io
dirname, _ = os.path.split(os.path.abspath(__file__))
default_folder = 'mylstm/'
def helper_saveparam(folder, params, prefix='', name=''):
    if not os.path.exists(folder):
        os.mkdir(folder)
    time1 = time.time()
    print(" ## save" , params)
    for param in params:
        np.savetxt(os.path.join(folder, prefix + name + param.name + '.npy'),
                   param.get_value(), fmt='%10.15f')
    time2 = time.time()
    print(" ## finsihed saving, takes time " , time2-time1)

# def load_label(name=''):
#     labels = pkl.load(open(dirname+'/' + default_folder + 'y_' + name + 'label.p', "rb"))
#     return labels
