from .base import *
import os

DEBUG = False

# ALLOWD HOST
ALLOWED_HOSTS = ['*']

# CORS
CORS_ORIGIN_ALLOW_ALL=True
CORS_ALLOW_CREDENTIALS = True
CORS_ORIGIN_WHITELIST = [
    'localhost',
    '127.0.0.1'
]