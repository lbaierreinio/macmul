from Crypto.Cipher import AES
from Crypto.Hash import HMAC, SHA256
from Crypto.Random import get_random_bytes

class MAC:
    def __init__(self):
        self.hmac_key = get_random_bytes(16)
        self.hmac = HMAC.new(self.hmac_key, digestmod=SHA256)
        self.tag = None

    def store_tag(self, params):
        str_params = str(params).encode()
        self.tag = self.hmac.update(str_params).digest()
        
    def verify(self, params):
        try: 
            hmac_r = HMAC.new(self.hmac_key, digestmod=SHA256)
            str_params = str(params).encode()
            tag = hmac_r.update(str_params).verify(self.tag)
        except ValueError:
            return False
        return True
    