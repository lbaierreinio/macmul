import argparse
from Crypto.Cipher import AES
from Crypto.Hash import HMAC, SHA256
from Crypto.Random import get_random_bytes


'''
USAGE: python3 ./ex_weights.py (0/1)
'''
def main():
    parser = argparse.ArgumentParser(description="Check your lab code")
    parser.add_argument('fail', type=str, help="Enter 1 if you want to fail the test, 0 o/w")
    args = parser.parse_args()

    if args.fail not in ['0', '1']:
        print("Incorrect arguments. Please enter 0 or 1")
        exit(-1)

    # Encode data into bytes
    weights = [0.1, 0.2, 0.3, 0.4]

    str_weights = str(weights).encode()

    hmac_key = get_random_bytes(16)

    # Get HMAC & HMAC digest (SHA256 specifies the hash function).
    hmac = HMAC.new(hmac_key, digestmod=SHA256)

    # Hash that can be compared against the receiver's hash
    tag = hmac.update(str_weights).digest()
    if args.fail == '1':
        tag = tag + b'bogus'

    # Check hash
    try: 
        hmac_r = HMAC.new(hmac_key, digestmod=SHA256)
        tag = hmac_r.update(str_weights).verify(tag)
        print('MAC verification passed')
    except ValueError:
        print('MAC verification failed')
        return

if __name__ == '__main__':
    main()