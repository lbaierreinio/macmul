import os
from dotenv import load_dotenv

def get_secret_key():
    load_dotenv()
    return str(os.getenv("SECRET_KEY")).encode('ascii')

def get_mac_prob():
    load_dotenv()
    return float(os.getenv("MAC_PROB"))