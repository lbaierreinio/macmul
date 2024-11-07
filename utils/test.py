from Crypto.Cipher import AES
from Crypto.Hash import HMAC, SHA256
from Crypto.Random import get_random_bytes
import numpy as np

def test(param): # param should be grid
    param = np.array([[1, 2, 3], [4, 5, 6]])
    hmac = HMAC.new(b'key', digestmod=SHA256)
    def blah(p, i):
        digest = hmac.update(str(p).encode()).digest()
        x = np.frombuffer(digest, dtype=np.uint64)
        return x[i]
        
    to_stack = []
    vectorized_func = np.vectorize(blah)
    for i in range(0, 4):
        to_stack.append(vectorized_func(param, i))
        
    return param, np.stack(to_stack)


    # return np.stack((param, vectorized_func(param)))