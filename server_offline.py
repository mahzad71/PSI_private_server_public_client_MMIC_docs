import pandas as pd
import pickle
#@@from oprf import client_prf_offline, order_of_generator, G
from time import time

# Client's PRF secret key (a value from range(order_of_generator))
#@@oprf_client_key = 12345678910111213141516171819222222222222
t0 = time()

# Key * generator of elliptic curve
#@@client_point_precomputed = (oprf_client_key % order_of_generator) * G

# Load the CSV file and extract the 'encoded_icd_codes' column
df = pd.read_csv('server_dataset_NC.csv')
encoded_icd_codes = df['encoded_icd_codes'].tolist()  # Using the numeric encoded values directly

# OPRF layer: encode the client's set as elliptic curve points
encoded_server_set = []
for code in encoded_icd_codes:
    encoded_server_set.append(code) 
# Save the encoded data
with open('server_preprocessed', 'wb') as g:
    pickle.dump(encoded_server_set, g)

t1 = time()
print('Server OFFLINE time: {:.2f}s'.format(t1-t0))
