import os
from dotenv import load_dotenv

def get_secret_key():
    load_dotenv()
    return str(os.getenv("SECRET_KEY")).encode('ascii')