import tenseal as ts
from time import time
import socket
import pickle
from math import log2
from parameters import sigma_max, output_bits, plain_modulus, poly_modulus_degree, number_of_hashes, bin_capacity, alpha, ell, hash_seeds
from cuckoo_hash import reconstruct_item, Cuckoo
from auxiliary_functions import windowing
from oprf import order_of_generator, client_prf_online_parallel
import logging
import pdb
import base64
import json
import pandas as pd
import numpy as np

oprf_client_key = 12345678910111213141516171819222222222222

log_no_hashes = int(log2(number_of_hashes)) + 1
base = 2 ** ell
minibin_capacity = int(bin_capacity / alpha)
logB_ell = int(log2(minibin_capacity) / ell) + 1 # <= 2 ** HE.depth
dummy_msg_client = 2 ** (sigma_max - output_bits + log_no_hashes)

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('localhost', 4470))

#------------------------------------------------------------------------Step1: Getting public key from server---------------------------------------------
try:
# The client receives bytes that represent the public HE context
    #!!pdb.set_trace()
    L = client.recv(10).decode().strip()
    L = int(L, 10)
    public_key = b""
    while len(public_key) < L:
        data = client.recv(4096)
        if not data:
            logging.error("No more data received for public key.")
            break
        public_key += data
    logging.debug(f"Received public key bytes: {public_key[:50]}...")  # Log first 50 bytes for inspection
    logging.info("Public key received from server.")

    t2 = time()    
    # Here we recover the public context received from the server
    received_data = pickle.loads(public_key)
    srv_context = ts.context_from(received_data)
    #pdb.set_trace()
    logging.info("Recovered the public context received from the server.")
    logging.debug(f"Deserialized public key type: {type(received_data)}")
    logging.debug(f"Deserialized public key content (sample): {str(received_data)[:200]}")
except Exception as e:
    logging.error(f"Error during public key reception or deserialization: {e}")
    # Decide whether to exit, retry, or handle error differently
    exit(1)

#-------------------------------------------------------------------Step2: OPRF client online-----------------------------------------------------------------

# We prepare the partially OPRF processed database to be sent to the server
#pdb.set_trace()
#@@pickle_off = open("client_preprocessed", "rb")
#@@encoded_client_set = pickle.load(pickle_off)
#@encoded_client_set_serialized = pickle.dumps(encoded_client_set, protocol=None)

#@@L = len(encoded_client_set_serialized)
#@@sL = str(L) + ' ' * (10 - len(str(L)))
#@@client_to_server_communiation_oprf = L #in bytes
# The length of the message is sent first
#@@client.sendall((sL).encode())
#@@client.sendall(encoded_client_set_serialized)

#@@L = client.recv(10).decode().strip()
#@@L = int(L, 10)

#@@PRFed_encoded_client_set_serialized = b""
#@@while len(PRFed_encoded_client_set_serialized) < L:
    #@@data = client.recv(4096)
    #@@if not data: break
    #@@PRFed_encoded_client_set_serialized += data   
#@@PRFed_encoded_client_set = pickle.loads(PRFed_encoded_client_set_serialized)
#@@t0 = time()
#@@server_to_client_communication_oprf = len(PRFed_encoded_client_set_serialized)

# We finalize the OPRF processing by applying the inverse of the secret key, oprf_client_key
#@@key_inverse = pow(oprf_client_key, -1, order_of_generator)
#@@PRFed_client_set = client_prf_online_parallel(key_inverse, PRFed_encoded_client_set)
#@@print(' * OPRF protocol done!')

#---------------------------------------------------------Step2: Reading the coeffes from client offline phase and then trnsposed and serialize them---------------------------------------------------------
g = open('client_preprocessed', 'rb')
poly_coeffs = pickle.load(g)

# For the online phase of the server, we need to use the columns of the preprocessed database
transposed_poly_coeffs = np.transpose(poly_coeffs).tolist()
print("Transposed coeffs before encryption:", transposed_poly_coeffs)

# public_context` is defined and is the appropriate context for encryption
serialized_enc_transposed_poly_coeffs = []
for transposed_poly_coeff in transposed_poly_coeffs:
    enc_transposed_poly_coeff = ts.bfv_vector(srv_context, transposed_poly_coeff)
    serialized_enc_transposed_poly_coeffs.append(enc_transposed_poly_coeff.serialize())

# Writing serialized encrypted coefficients to a file using pickle
with open('encrypted_transposed_coeffs.pkl', 'wb') as f:
    pickle.dump(serialized_enc_transposed_poly_coeffs, f)

print("Serialized and encrypted coefficients have been written to 'encrypted_transposed_coeffs.pkl'")


#------------------------------------------------------Step6: Sending encrypted data to server-----------------------------------------------------------------------------------
pdb.set_trace()
context_serialized = srv_context.serialize()
message_to_be_sent = [context_serialized, serialized_enc_transposed_poly_coeffs]
message_to_be_sent_serialized = pickle.dumps(message_to_be_sent, protocol=None)
t1 = time()
L = len(message_to_be_sent_serialized)
sL = str(L) + ' ' * (10 - len(str(L)))
client_to_server_communiation_query = L 
#the lenght of the message is sent first
client.sendall((sL).encode())
print(" * Sending the context and ciphertext to the server....")
# Now we send the message to the server
client.sendall(message_to_be_sent_serialized)

#------------------------------------------------------------Step7: receive and Decrypt the results of PSI frpm server--------------------------------------------------
pdb.set_trace()
print(" * Waiting for the servers's answer...")

# The answer obtained from the server:
L = client.recv(10).decode().strip()
L = int(L, 10)
answer = b""
while len(answer) < L:
    data = client.recv(4096)
    if not data: break
    answer += data
t2 = time()
server_to_client_query_response = len(answer) #bytes
# Here is the vector of decryptions of the answer
intersected_codes  = pickle.loads(answer)
#decryptions = []
#for ct in ciphertexts:
    #decryptions.append(ct)

# Load the client_dataset_NC.csv to find file numbers
df = pd.read_csv('client_dataset_NC.csv')

results = df[df['unique_number'].isin(intersected_codes)]

# Group results by 'file_number' and aggregate the 'icd_code' in a list
grouped_results = results.groupby('file_number').agg({
    'icd_code': lambda x: list(x)  # This will collect all icd_codes into a list for each file_number
}).reset_index()

# Output the results
print("File numbers with their corresponding ICD codes in the intersection:")
for index, row in grouped_results.iterrows():
    print(f"File Number: {row['file_number']}, ICD Codes: {row['icd_code']}")

all_icd_codes_in_intersection = results['icd_code'].unique()
print("All ICD Codes in the Intersection:", all_icd_codes_in_intersection)


#----------------------------------------------------------Step9: heck the matches of PSIs with the real intesection we had----------------------------------------------------
#!!pdb.set_trace()
df_intersection = pd.read_csv('intersection_dataset.csv')
real_intersection = df_intersection['unique_number'].tolist()
t3 = time()
print('\n Intersection recovered correctly: {}'.format(set(intersected_codes) == set(real_intersection)))
print("Client set intersection is:", intersected_codes)
print("Real intersection is:", real_intersection)

#print("Subject IDs of intersection:", client_intersection_subject_ids)
#print("ICD codes of intersection:", client_intersection_icd_code)

print("Disconnecting...\n")
#print('  Client ONLINE computation time {:.2f}s'.format(t1 - t0 + t3 - t2))
print('  Communication size:')
#@@print('    ~ Client --> Server:  {:.2f} MB'.format((client_to_server_communiation_oprf + client_to_server_communiation_query )/ 2 ** 20))
#@@print('    ~ Server --> Client:  {:.2f} MB'.format((server_to_client_communication_oprf + server_to_client_query_response )/ 2 ** 20))
client.close()


