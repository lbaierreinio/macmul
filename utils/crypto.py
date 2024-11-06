from Crypto.Cipher import AES
from Crypto.Hash import HMAC, SHA256
from Crypto.Random import get_random_bytes

def cu_hash(self, hmac, x):
       return hmac.update(str(x).encode()).digest()