from __future__ import division

import argparse
from collections import Counter
import copy
import csv
import json
from math import log2
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import trange

def get_alphabet():
    with open('alphabet.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            alphabet = row

    return alphabet

def get_text(fname='ciphertext.txt', as_array=True):
    with open(fname, 'r') as f:
        text = f.read()

    text = text.replace('\n', ' ')

    if as_array:
        text = list(text)

    return text

def char_to_index(text, A):
    char_index_mapping = {char: index for index, char in enumerate(A)}
    indices = [char_index_mapping[char] for char in text]

    return indices

def index_to_char(indices, A):
    return [A[index] for index in indices]

def get_letter_probabilities():
    with open('letter_probabilities.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            P = [float(p) for p in row]

    return np.array(P)

def get_letter_transition_matrix():
    with open('letter_transition_matrix.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        M = []
        for row in reader:
            M.append([float(p) for p in row])

    return np.array(M)

def random_permutation_f(array):
    perm = np.random.permutation(array)
    f = {array[i]: perm[i] for i in range(len(array))}

    return f

def encrypt(text, A):
    f = random_permutation_f(A)
    ciphertext = [f[char] for char in text]

    return ciphertext

def invert_f(f):
    f_inverse = {v: k for k, v in f.items()}

    return f_inverse

def decrypt(f, y):
    f_inverse = invert_f(f)
    x_hat = [f_inverse[y_i] for y_i in y]

    return x_hat

def p_f_y_tilde(f, y, P, M):
    # Note: equal to p_y_f

    f_inverse = invert_f(f)
    x_hat = decrypt(f, y)
    p = P[x_hat[0]] * np.prod([M[x_hat[i], x_hat[i-1]] for i in range(1, len(x_hat))])

    return p

def log_p_f_y_tilde(f, y, P, M):
    # Note: equal to log(p_y_f)

    f_inverse = invert_f(f)
    x_hat = decrypt(f, y)
    log_p = np.log2(P[x_hat[0]])
    log_p += np.sum(np.log2([M[x_hat[i], x_hat[i-1]] for i in range(1, len(x_hat))]))

    return log_p

def acceptance_prob(f, f_prime, y, P, M):
    # p_f_prime / p_f = exp(log(p_f_prime)) / exp(log(p_f))
    # = exp(log(p_f_prime) - log(p_f))

    log_p_f_y_tilde_f_prime = log_p_f_y_tilde(f_prime, y, P, M)
    log_p_f_y_tilde_f = log_p_f_y_tilde(f, y, P, M)
    ratio = np.exp(log_p_f_y_tilde_f_prime - log_p_f_y_tilde_f)

    a = min(1, ratio)

    return a

def most_common(lst):
    counter = Counter(lst)
    most_common_element, count = counter.most_common()[0]

    return most_common_element

def mcmc(A, y, P, M, num_iters):
    indices = list(range(len(A)))
    fs = []
    f = random_permutation_f(indices)  # random initialization

    for _ in trange(num_iters):
        index1, index2 = np.random.choice(indices, size=2, replace=False)
        f_prime = copy.deepcopy(f)
        f_prime[index1], f_prime[index2] = f_prime[index2], f_prime[index1]

        a = acceptance_prob(f, f_prime, y, P, M)

        if np.random.rand() <= a:
            f = f_prime

        fs.append(str(f))

    f_star = eval(most_common(fs))
    log_likelihood = log_p_f_y_tilde(f, y, P, M)

    return f_star, log_likelihood

def multi_mcmc(A, y, P, M, num_iters, num_mcmcs):
    best_f = None
    best_log_likelihood = float('-inf')

    for _ in trange(num_mcmcs):
        f, log_likelihood = mcmc(A, y, P, M, num_iters)
        print(log_likelihood)

        if log_likelihood >= best_log_likelihood:
            best_f = f
            best_log_likelihood = log_likelihood

    return best_f

def main(custom_encrypt, training_size, num_iters, num_mcmc):
    A = get_alphabet()
    if custom_encrypt:
        ciphertext = encrypt(get_text('plaintext.txt'), A)
    else:
        ciphertext = get_text('ciphertext.txt')
    training_ciphertext = ciphertext[:training_size]
    y = char_to_index(training_ciphertext, A)
    P = get_letter_probabilities()
    M = get_letter_transition_matrix()

    f_star = multi_mcmc(A, y, P, M, num_iters, num_mcmc)

    y_full = char_to_index(ciphertext, A)
    x_hat = decrypt(f_star, y_full)
    plaintext_hat = index_to_char(x_hat, A)
    print(''.join(plaintext_hat))

    plaintext = get_text('plaintext.txt')
    print('Accuracy = {}'.format(accuracy_score(plaintext, plaintext_hat)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--custom_encrypt', action='store_true', default=False,
        help='Whether to use a custom encryption permutation instead of the default')
    parser.add_argument('--training_size', type=int, default=1000,
        help='Number of characters to use when learning decryption function')
    parser.add_argument('--num_iters', type=int, default=10000,
        help='Number of iterations in each MCMC run')
    parser.add_argument('--num_mcmc', type=int, default=10,
        help='Number of times to run MCMC algorithm')
    args = parser.parse_args()

    main(args.custom_encrypt, args.training_size, args.num_iters, args.num_mcmc)
