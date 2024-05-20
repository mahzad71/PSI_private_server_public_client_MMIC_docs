import pandas as pd
from parameters import sigma_max, number_of_hashes, output_bits, bin_capacity, alpha, hash_seeds, plain_modulus
from simple_hash import Simple_hash
from auxiliary_functions import coeffs_from_roots
import pickle
#@@from oprf import server_prf_offline_parallel, order_of_generator, G
from math import log2
from time import time

# Load the data from a CSV file
df = pd.read_csv('client_dataset_NC.csv')

# Use the pre-encoded 'encoded_icd_codes' directly from the dataset
client_set = df['unique_number'].tolist()

# server's PRF secret key
#@@oprf_server_key = 1234567891011121314151617181920

# key * generator of elliptic curve
#@@server_point_precomputed = (oprf_server_key % order_of_generator) * G

t0 = time()
# The PRF function is applied on the set of the server, using parallel computation
#@@PRFed_server_set = server_prf_offline_parallel(server_set, server_point_precomputed)
PRFed_client_set = set(client_set)
#@@t1 = time()

log_no_hashes = int(log2(number_of_hashes)) + 1
dummy_msg_client = 2 ** (sigma_max - output_bits + log_no_hashes) + 1 
client_size = len(client_set)
minibin_capacity = int(bin_capacity / alpha)
number_of_bins = 2 ** output_bits

# The OPRF-processed database entries are simple hashed
SH = Simple_hash(hash_seeds)
for item in PRFed_client_set:
    for i in range(number_of_hashes):
        SH.insert(item, i)

# simple_hashed_data is padded with dummy_msg_server
for i in range(number_of_bins):
    for j in range(bin_capacity):
        if SH.simple_hashed_data[i][j] == None:
            SH.simple_hashed_data[i][j] = dummy_msg_client

# Here we perform the partitioning:
# Namely, we partition each bin into alpha minibins with B/alpha items each
# We represent each minibin as the coefficients of a polynomial of degree B/alpha that vanishes in all the entries of the mininbin
# Therefore, each minibin will be represented by B/alpha + 1 coefficients; notice that the leading coeff = 1
#@@t2 = time()

poly_coeffs = []
for i in range(number_of_bins):
    # we create a list of coefficients of all minibins from concatenating the list of coefficients of each minibin
    coeffs_from_bin = []
    for j in range(alpha):
        roots = [SH.simple_hashed_data[i][minibin_capacity * j + r] for r in range(minibin_capacity)]
        coeffs_from_bin = coeffs_from_bin + coeffs_from_roots(roots, plain_modulus).tolist()
    poly_coeffs.append(coeffs_from_bin)

f = open('client_preprocessed', 'wb')
pickle.dump(poly_coeffs, f)
f.close()
t3 = time()
#print('OPRF preprocessing time {:.2f}s'.format(t1 - t0))
#print('Hashing time {:.2f}s'.format(t2 - t1))
#print('Poly coefficients from roots time {:.2f}s'.format(t3 - t2))
print('Client OFFLINE time {:.2f}s'.format(t3 - t0))
