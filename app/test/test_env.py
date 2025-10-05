# test_env.py
import os
from dotenv import load_dotenv

load_dotenv()

print("MAIL USER:", os.getenv("MAIL_USERNAME"))