import socket
import tenseal as ts
import pickle
import numpy as np
from math import log2
import logging
import pdb
import json
import pandas as pd

from parameters import sigma_max, output_bits, plain_modulus, poly_modulus_degree, number_of_hashes, bin_capacity, alpha, ell, hash_seeds
from cuckoo_hash import reconstruct_item, Cuckoo
from auxiliary_functions import windowing

from parameters import number_of_hashes, bin_capacity, alpha, ell, plain_modulus, poly_modulus_degree
from auxiliary_functions import power_reconstruct
from oprf import server_prf_online_parallel

oprf_server_key = 1234567891011121314151617181920
from time import time

log_no_hashes = int(log2(number_of_hashes)) + 1
base = 2 ** ell
minibin_capacity = int(bin_capacity / alpha)
logB_ell = int(log2(minibin_capacity) / ell) + 1 # <= 2 ** HE.depth
dummy_msg_server = 2 ** (sigma_max - output_bits + log_no_hashes)

# Initialize and listen on the server socket
serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #serv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    serv.bind(('localhost', 4470))
    serv.listen(1)
    logging.info("Server is listening...")
except Exception as e:
    logging.error(f"Error setting up server socket: {e}")
    exit(1)

#------------------------------Step0: Setting the public and private contexts for the BFV Homorphic Encryption scheme-------------------------------
pdb.set_trace()
private_context = ts.context(ts.SCHEME_TYPE.BFV, poly_modulus_degree=poly_modulus_degree, plain_modulus=plain_modulus)
public_context = ts.context_from(private_context.serialize())
public_context.make_context_public()
    

for i in range(1):
    conn, addr = serv.accept()

#---------------------------------------------------------Step1: Sending the public context to client------------------------------------------------
    pdb.set_trace()
    try:
        # Serializing public context to be sent to the client
        context_serialized = public_context.serialize()
        #pdb.set_trace()
        context_serialized_to_be_sent = pickle.dumps(context_serialized, protocol=None)
        t1 = time()
        logging.info("Serializing public context to be sent to the client.")

        # Preparing length information
        L = len(context_serialized_to_be_sent)
        sL = str(L) + ' ' * (10 - len(str(L))) #pad len to 10 bytes
        public_key_communication = L 

        # Sending the length of the message first
        conn.sendall(sL.encode())
        # Sending the serialized public context
        conn.sendall(context_serialized_to_be_sent)
        #pdb.set_trace()
        logging.info("Sending the public key context to the client....")
        t3 = time()

    except Exception as e:
        logging.error(f"Error sending public context to client: {e}")

#----------------------------------------------------Step2: OPRF online in server for client's data------------------------------------------------------
    #!!pdb.set_trace()
    #@@L = conn.recv(10).decode().strip()
    #@@L = int(L, 10)
    # OPRF layer: the server receives the encoded set elements as curve points
    #@@encoded_client_set_serialized = b""
    #@@while len(encoded_client_set_serialized) < L:
        #@@data = conn.recv(4096)
        #@@if not data: break
        #@@ed_client_set = pickle.loads(encoded_client_set_serialized)
    #@@t0 = time()
    # The server computes (parallel computation) the online part of the OPRF protocol, using its own secret key
    #@@PRFed_encoded_client_set = server_prf_online_parallel(oprf_server_key, encoded_client_set)
    #@@PRFed_encoded_client_set_serialized = pickle.dumps(PRFed_encoded_client_set, protocol=None)
    #@@L = len(PRFed_encoded_client_set_serialized)
    #@@sL = str(L) + ' ' * (10 - len(str(L))) #pad len to 10 bytes

    #@@conn.sendall((sL).encode())
    #@@conn.sendall(PRFed_encoded_client_set_serialized)    
    #@@print(' * OPRF layer done!')
    #@@t1 = time()

    #-------------------------------------------------------Step3: Getting encrypted data from client and recover the context---------------------------------------------------------
    pdb.set_trace()
    L = conn.recv(10).decode().strip()
    L = int(L, 10)

    # The server receives bytes that represent the public HE context and the query ciphertext
    final_data = b""
    while len(final_data) < L:
        data = conn.recv(4096)
        if not data: break
        final_data += data

    #@@t2 = time()    
    # Here we recover the context and ciphertext received from the received bytes
    received_data = pickle.loads(final_data)
    srv_context = ts.context_from(received_data[0])
    received_enc_query_serialized = received_data[1]

    enc_client_coeffs = []
    for item in received_enc_query_serialized:
        enc_vector = ts.bfv_vector_from(private_context, item)
        enc_client_coeffs.append(enc_vector)
        print(f"Received clients data in server side", enc_vector)

    #--------------------------------------------------Step3: Insert server data in the hash function------------------------------------------------

    df = pd.read_csv('server_dataset_NC.csv')

    # Extract the 'encoded_icd_codes' column
    server_set = df['unique_number'].tolist()

    # Encrypt each unique ICD code
    encrypted_icd_codes = []
    for index, row in df.iterrows():
        icd_code = int(row['encoded_icd_codes'])  # Convert ICD code to integer if necessary
        encrypted_code = ts.bfv_vector(private_context, [icd_code])
        encrypted_icd_codes.append(encrypted_code.serialize())

        print("encoded_icd_codes", encrypted_code)  # Print the TenSEAL encrypted vector object

    # Save the serialized encrypted ICD codes
    with open('encrypted_icd_codes.pkl', 'wb') as f:
        pickle.dump(encrypted_icd_codes, f)

    print("Encrypted ICD codes have been serialized and saved to 'encrypted_icd_codes.pkl'")
    

    # Each item from the client set is mapped to a Cuckoo hash table
    pdb.set_trace()
    CH = Cuckoo(hash_seeds)
    for item in server_set:
        CH.insert(item)

    # We padd the Cuckoo vector with dummy messages
    for i in range(CH.number_of_bins):
        if (CH.data_structure[i] == None):
            CH.data_structure[i] = dummy_msg_server

    #-----------------------------------------------------Step4: Windowing on hashed server's data---------------------------------------------------------------------

    # We apply the windowing procedure for each item from the Cuckoo structure
    pdb.set_trace()
    windowed_items = []
    for item in CH.data_structure:
        windowed_items.append(windowing(item, minibin_capacity, plain_modulus))


    #---------------------------------------------------------------Step5: Batching and encryption on server's data--------------------------------------------------------------

    plain_server_data = [None for k in range(len(windowed_items))]
    enc_server_data = [[None for j in range(logB_ell)] for i in range(1, base)]
    # We create the <<batched>> query to be sent to the server
    # By our choice of parameters, number of bins = poly modulus degree (m/N =1), so we get (base - 1) * logB_ell ciphertexts
    #!!pdb.set_trace()
    for j in range(logB_ell):
        for i in range(base - 1):
            if ((i + 1) * base ** j - 1 < minibin_capacity):
                for k in range(len(windowed_items)):
                    plain_server_data[k] = windowed_items[k][i][j]
                enc_server_data[i][j] = ts.bfv_vector(private_context, plain_server_data)

    print("Encrypted formmat of server's data with server's private key :", enc_server_data)
    

   #----------------------------------------------------------------Step4: Recover encrypted powers-----------------------------------------------------------------------------
    pdb.set_trace()
    # Here we recover all the encrypted powers Enc(y), Enc(y^2), Enc(y^3) ..., Enc(y^{minibin_capacity}), from the encrypted windowing of y.
    # These are needed to compute the polynomial of degree minibin_capacit
    all_powers = [None for i in range(minibin_capacity)]
    for i in range(base - 1):
        for j in range(logB_ell):
            if ((i + 1) * base ** j - 1 < minibin_capacity):
                all_powers[(i + 1) * base ** j - 1] = enc_server_data[i][j]
    
    for k in range(minibin_capacity):
        if all_powers[k] == None:
            all_powers[k] = power_reconstruct(enc_server_data, k + 1)
    all_powers = all_powers[::-1]
    print("All powers results:", all_powers)    
    
#----------------------------------------------------------Step5: Dot product between Coeffs and server's data---------------------------------------------------

    # Server sends alpha ciphertexts, obtained from performing dot_product between the polynomial coefficients from the preprocessed server database and all the powers Enc(y), ..., Enc(y^{minibin_capacity})
    pdb.set_trace()
    srv_answer = []
    srv_answer_serialized = []
    #for p_value in decrypted_value:
    for i in range(alpha):
        # the rows with index multiple of (B/alpha+1) have only 1's
        dot_product = all_powers[0]
        for j in range(1, minibin_capacity):
            dot_product = dot_product + enc_client_coeffs[(minibin_capacity + 1) * i + j] * all_powers[j]
        dot_product = dot_product + enc_client_coeffs[(minibin_capacity + 1) * i + minibin_capacity]
        print("Dot product is:", dot_product)
        srv_answer.append(dot_product)
    for product_value in srv_answer:
        srv_answer_serialized.append(product_value.serialize())
        # Here is the vector of decryptions of the answer
        
        #ciphertexts = pickle.loads(srv_answer)
    decryptions = []
    for ct in srv_answer_serialized:
        decryptions.append(ts.bfv_vector_from(private_context, ct).decrypt())
    print("Decrypted intersection dot prodoct is:", decryptions)
    # Write the decrypted data to a file
    with open('Decrypted_intersection_dotprodoct.txt', 'w') as file:
        for row in decryptions:
            line = ', '.join(str(x) for x in row)  # This will convert each item in the row to a string, skipping 'None' values
            file.write(line + '\n')
    # Counting zeros in decrypted dot products
    zero_count = sum(1 for row in decryptions for item in row if item == 0)
    print("Number of zeros in decrypted dot products:", zero_count)
    
#---------------------------------------------Stp8: Recoverig and Finding the location of intersection in server set-----------------------------------------------------------------
pdb.set_trace()
recover_CH_structure = []
for matrix in windowed_items:
    recover_CH_structure.append(matrix[0][0])

count = [0] * alpha

# Load the CSV file
df = pd.read_csv('server_dataset_NC.csv')
server_set_entries = df['unique_number'].tolist()

server_intersection = []


print("Size of recover_CH_structure:", len(recover_CH_structure))
print("poly_modulus_degree:", poly_modulus_degree)
print("Size of hash_seeds:", len(hash_seeds))

import math

for j in range(alpha):
    for i in range(poly_modulus_degree):
        if decryptions[j][i] == 0:
            count[j] = count[j] + 1
            hash_index = recover_CH_structure[i] % (2 ** log_no_hashes)
            if hash_index >= len(hash_seeds):
                print(f"Error: Calculated hash seed index {hash_index} is out of range for hash_seeds.")
            else:
                server_common_element = reconstruct_item(recover_CH_structure[i], i, hash_seeds[hash_index])
            if server_common_element is not None:
                print(f"Trying to find reconstructed element {server_common_element} in server set...")
                if server_common_element in server_set:
                    index = server_set.index(server_common_element)
                    server_intersection.append(server_set_entries[index])
                else:
                    print(f"Element {server_common_element} not found in server's encoded set.")
            else:
                print(f"Reconstruction failed for index {i}")


with open('intersection_subject_ids.txt', 'w') as file:
    for subject_id in server_intersection:
        file.write(str(subject_id) + '\n')

#--------------------------------------------------------Step6: Sending answers to the client----------------------------------------------------------------------

    # The answer to be sent to the client is prepared
    pdb.set_trace()
    response_to_be_sent = pickle.dumps(server_intersection, protocol=None)
    t3 = time()
    L = len(response_to_be_sent)
    sL = str(L) + ' ' * (10 - len(str(L))) #pad len to 10 bytes

    conn.sendall((sL).encode())
    conn.sendall(response_to_be_sent)

    # Close the connection
    print("Client disconnected \n")
    #@@print('Server ONLINE computation time {:.2f}s'.format(t1 - t0 + t3 - t2))

    conn.close()
