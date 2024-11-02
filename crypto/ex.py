from Crypto.Cipher import AES
from Crypto.Hash import HMAC, SHA256
from Crypto.Random import get_random_bytes

def main():
    '''
    Sender side
    '''
    # Encode data into bytes
    data = 'secret data to transmit'.encode()

    # Get AES key & HMAC key using random bytes.
    aes_key = get_random_bytes(16)
    hmac_key = get_random_bytes(16)

    # Get cipher & cipher text.
    # Clearly, we need to make sure we encrypt the text,
    # or else there is no point in using encryption.
    cipher = AES.new(aes_key, AES.MODE_CTR)
    ciphertext = cipher.encrypt(data)

    # Get HMAC & HMAC digest (SHA256 specifies the hash function).
    hmac = HMAC.new(hmac_key, digestmod=SHA256)

    # This is the tag that is sent to the receiver
    # such that it can be verified as the MAC
    tag = hmac.update(ciphertext).digest() + b'bogus'

    '''
    Receiver side
    '''
    try: 
        hmac_r = HMAC.new(hmac_key, digestmod=SHA256)
        tag = hmac_r.update(ciphertext).verify(tag)
        print('MAC verification passed')
    except ValueError:
        print('MAC verification failed')
        return

if __name__ == '__main__':
    main()