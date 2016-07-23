# -*- coding: utf-8 -*-
# Python 2.7.11 :: Anaconda

from __future__ import print_function
import numpy as np


num_of_reviewer = 943
num_of_item = 1682

def loadMovie(filename = None, path = './ml-100k/'):
    '''
    this function returns dictionary of ratings of movies by users.
    Key is user_number(str) and Value.
    Value contains dictionary whose Key is movies' title and Value is rating.
    '''
    movies = makeMovieDictionary()

    if filename == None:
        filename = 'u.data'
    prefs = {}

    for line in open(path + filename, 'r'):
        (user, movie_id, rating, ts) = line.split('\t')
        prefs.setdefault(user, {})
        prefs[user][movies[movie_id]] = float(rating)

    return prefs

def convertDicToArray(prefs = None, cols = num_of_reviewer, rows = num_of_item):
    '''
    loadMovie関数で作った辞書をnumpy ndarrayに変換する
    '''
    R = np.zeros((cols, rows), dtype = np.float32)
    movies = makeMovieDictionary()

    if prefs == None:
        prefs = loadMovie()

    for person in prefs:
        for item in prefs[person]:
            item_index = int(movies.keys()[movies.values().index(item)]) - 1
            R[int(person) - 1, item_index] = prefs[person][item]

    return R

def prepare_nfm(k = 20):
    '''
    NFMの準備として特徴量をkとして正のランダムな値で初期化したP,Qを用意する
    '''
    P = np.random.rand(num_of_reviewer, k)
    Q = np.random.rand(num_of_item, k)

    return P, Q

def nmf(R, P = None, Q = None, epochs = 5000, alpha = 0.0002, beta = 0.02):

    cols = R.shape[0]
    rows = R.shape[1]

    if P is None and Q is None:
        P, Q = prepare_nfm()

    Q = Q.T

    K = P.shape[1]

    for epoch in range(epochs):
        e = .0
        for i in range(cols):
            for j in range(rows):
                if R[i, j] > 0:
                    eij = R[i, j] - np.dot(P[i,:], Q[:,j])
                    for k in range(K):
                        P[i, k] = P[i, k] + alpha * (2 * eij * Q[k, j] - beta * P[i, k])
                        Q[k,j] = Q[k,j] + alpha * (2 * eij * P[i,k] - beta * Q[k,j])

        for i in range(cols):
            for j in range(rows):
                if R[i, j] > 0:
                    e = e + pow(R[i,j] - np.dot(P[i,:], Q[:,j]), 2)

                    for k in range(K):
                        e = e + (beta / 2) * (pow(P[i,k],2) + pow(Q[k,j], 2))

        if epoch % 10 == 0:
            print("Error: {}".format(e))
            # 10回1回erroを表示


        if e < 0.001:
            break
            # 終了条件

    return P, Q.T

def main():
    prefs_1 = loadMovie(filename = 'u1.base')
    R_1 = convertDicToArray(prefs = prefs_1)
    P_1, Q_1 = prepare_nfm(k = 20)
    P_1_, Q_1_ = nmf(R = R_1, P = P_1, Q = Q_1, epochs = 50)
    R_1_hat = np.dot(P_1_, Q_1_.T)
    prefs_1_test = loadMovie(filename = 'u1.test')
