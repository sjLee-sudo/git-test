from .base import *
import os

DEBUG = True


INSTALLED_APPS += [
    'debug_toolbar',
    'graphene_django',
]

# graphql schema
GRAPHENE = {
    'SCHEMA': 'terms.schema.schema' # Where your Graphene schema lives
}

MIDDLEWARE += [
    'debug_toolbar.middleware.DebugToolbarMiddleware',
]

INTERNAL_IPS = [
    '127.0.0.1',
]

def show_toolbar(request):
    return True
DEBUG_TOOLBAR_CONFIG = {
    "SHOW_TOOLBAR_CALLBACK" : show_toolbar,
}



# ALLOWD HOST
ALLOWED_HOSTS = ['*']

# CORS
CORS_ORIGIN_ALLOW_ALL=True
CORS_ALLOW_CREDENTIALS = True