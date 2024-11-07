import numpy as np
from Crypto.Cipher import AES
from Crypto.Hash import HMAC, SHA256

def cu_hash(param, key):
    # Instantiate HMAC object
    hmac = HMAC.new(key, digestmod=SHA256)
    # Return 64-bit slice of hash at specified index
    def get_hash_at_index(p, i):
        digest = hmac.update(str(p).encode()).digest()
        return np.frombuffer(digest, dtype=np.uint64)[i]
    # Stack the 64-bit slices
    to_stack = []
    vectorized_func = np.vectorize(get_hash_at_index)
    for i in range(0, 4):
        to_stack.append(vectorized_func(param, i))
        
    return np.stack(to_stack)
