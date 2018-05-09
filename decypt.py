from __future__ import division

import argparse
from collections import Counter
import copy
import csv
import numpy as np
from tempfile import NamedTemporaryFile


# ALPHABET


def get_alphabet(fname='alphabet.csv'):
    with open(fname, 'r') as csvfile:
        alphabet = np.array(list(csv.reader(csvfile))[0])

    return alphabet


# TEXT MODIFICATION


def char_to_index(text, alphabet=None):
    alphabet = alphabet if alphabet is not None else get_alphabet()

    char_index_mapping = {char: index for index, char in enumerate(alphabet)}
    indices = [char_index_mapping[char] for char in text]

    return indices

def index_to_char(indices, alphabet=None):
    alphabet = alphabet if alphabet is not None else get_alphabet()

    text = [alphabet[index] for index in indices]

    return text


# DATA LOADING


def get_text(fname, indices=False):
    with open(fname, 'r') as f:
        text = np.array(list(f.read().replace('\n', ' ')))

    return char_to_index(text) if indices else text

def get_letter_probabilities(fname='letter_probabilities.csv'):
    with open(fname, 'r') as csvfile:
        P = np.array([[float(p) for p in row] for row in csv.reader(csvfile)][0])

    return P

def get_letter_transition_matrix(fname='letter_transition_matrix.csv'):
    with open(fname, 'r') as csvfile:
        M = np.array([[float(p) for p in row] for row in csv.reader(csvfile)])

    return M


# ENCRYPTION AND DECRYPTION


def encrypt_text(x, alphabet=None):
    alphabet = alphabet if alphabet is not None else get_alphabet()

    perm = np.random.permutation(alphabet)
    f = {alphabet[i]: perm[i] for i in range(len(alphabet))}
    y = [f[x_i] for x_i in x]

    return y

def decrypt_text(f, y):
    x_hat = [f[y_i] for y_i in y]

    return x_hat


# COUNTS CLASS


class Counts:
    def __init__(self, y, alphabet):
        self.y = y
        self.first = y[0]
        self.counts = np.zeros((len(alphabet), len(alphabet)))
        for i in range(1, len(y)):
            self.counts[y[i], y[i-1]] += 1

    def decrypt_counts(self, f):
        f_inverse = np.zeros(len(f), dtype=int)
        for index1, index2 in enumerate(f):
            f_inverse[index2] = index1
        decrypted_counts = self.counts[f_inverse]
        decrypted_counts = decrypted_counts[:, f_inverse]

        return decrypted_counts


# MARKOV CHAIN MONTE CARLO


def log_p_f_y_tilde(f, y_counts, log_P, log_M):
    # Note: equal to log(p_y_f)

    log_prob = log_P[f[y_counts.first]] + \
               np.sum(np.multiply(log_M,
                                  y_counts.decrypt_counts(f)))

    return log_prob

def acceptance_prob(f, f_prime, y_counts, log_P, log_M):
    # p_f_prime / p_f = exp(log(p_f_prime)) / exp(log(p_f))
    # = exp(log(p_f_prime) - log(p_f))

    log_p_f_y_tilde_f_prime = log_p_f_y_tilde(f_prime, y_counts, log_P, log_M)
    log_p_f_y_tilde_f = log_p_f_y_tilde(f, y_counts, log_P, log_M)
    ratio = np.exp(log_p_f_y_tilde_f_prime - log_p_f_y_tilde_f)

    a = min(1, ratio)

    return a

def most_common(lst):
    counter = Counter(lst)
    most_common_element = counter.most_common()[0][0]

    return most_common_element

def mcmc(alphabet, y_counts, log_P, log_M, num_iters):
    fs = []
    f = np.random.permutation(len(alphabet))  # random initialization

    for _ in range(num_iters):
        index1, index2 = np.random.choice(len(alphabet), size=2, replace=False)
        f_prime = copy.deepcopy(f)
        f_prime[index1], f_prime[index2] = f_prime[index2], f_prime[index1]

        a = acceptance_prob(f, f_prime, y_counts, log_P, log_M)

        if np.random.rand() <= a:
            f = f_prime

        fs.append(str(f.tolist()))

    f_star = np.array(eval(most_common(fs)))
    log_likelihood = log_p_f_y_tilde(f_star, y_counts, log_P, log_M)

    return f_star, log_likelihood

def multi_mcmc(alphabet, y, training_size, log_P, log_M, num_iters, num_mcmcs):
    start = np.random.randint(max(0, len(y)-training_size) + 1)
    training_y = y[start:start+training_size]
    y_counts = Counts(training_y, alphabet)

    best_f = None
    best_log_likelihood = float('-inf')

    for _ in range(num_mcmcs):
        f, log_likelihood = mcmc(alphabet, y_counts, log_P, log_M, num_iters)

        if log_likelihood >= best_log_likelihood:
            best_f = f
            best_log_likelihood = log_likelihood

    return best_f


# EVALUATION


def accuracy_score(true, pred):
    true = np.array(list(true))
    pred = np.array(list(pred))
    accuracy = np.sum(true == pred) / len(true)
    
    return accuracy


# DECRYPT


def decrypt(ciphertext, output_file_name,
            training_size=1000, num_iters=10000, num_mcmc=10):
    # Load alphabet and convert text to indices
    alphabet = get_alphabet()
    y = char_to_index(ciphertext)

    # Load probabilities and convert to log domain
    log_P = np.log(get_letter_probabilities())
    log_M = np.nan_to_num(np.log(get_letter_transition_matrix()))

    # Run MCMC
    f_star = multi_mcmc(alphabet, y, training_size, log_P, log_M, num_iters, num_mcmc)

    # Decrypt full ciphertext
    x_hat = decrypt_text(f_star, y)
    plaintext_hat = ''.join(index_to_char(x_hat, alphabet))

    # Write decrypted ciphertext
    with open(output_file_name, 'w') as f:
        f.write(plaintext_hat)

    return plaintext_hat


# MAIN


def main(plaintext_name, training_size, num_iters, num_mcmc):
    # Get ciphertext
    ciphertext = encrypt_text(get_text(plaintext_name))

    # Run decrpytion
    with NamedTemporaryFile() as f:
        plaintext_hat = decrypt(ciphertext, f.name, training_size, num_iters, num_mcmc)
        print(plaintext_hat)

    # Compare decryption to plaintext
    plaintext = get_text(plaintext_name)
    print('Accuracy = {}'.format(accuracy_score(plaintext, plaintext_hat)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plaintext_name', type=str, default='plaintext.txt',
        help='Name of plaintext file')
    parser.add_argument('--training_size', type=int, default=1000,
        help='Number of characters to use when learning decryption function')
    parser.add_argument('--num_iters', type=int, default=10000,
        help='Number of iterations in each MCMC run')
    parser.add_argument('--num_mcmc', type=int, default=10,
        help='Number of times to run MCMC algorithm')
    args = parser.parse_args()

    main(args.plaintext_name, args.training_size, args.num_iters, args.num_mcmc)