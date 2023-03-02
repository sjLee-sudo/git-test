from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
import json
import base64

if __name__ != '__main__':
    from django.conf import settings
    from django.db.models import Q

class EnDecrypt:
    def __init__(self, key):
        self.key = key
        self.f = Fernet(self.key)

    def encrypt(self, data, is_out_string=True):
        if isinstance(data, bytes):
            ou = self.f.encrypt(data)
        else:
            ou = self.f.encrypt(data.encode("utf-8"))
        if is_out_string is True:
            return ou.decode("utf-8")
        else:
            return ou

    def decrypt(self, data, is_out_string=True):
        if isinstance(data, bytes):
            ou = self.f.decrypt(data)
        else:
            ou = self.f.decrypt(data.encode("utf-8"))
        if is_out_string is True:
            return ou.decode("utf-8")
        else:
            return ou

class TempDatabaseInfoEnDecrypt():
    def __init__(self):
        temp_pw = 'asdf'
        self.derived_key = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=None,
            backend=default_backend()
        ).derive(bytes(temp_pw,encoding='utf-8'))
        self.encrypt_key = base64.urlsafe_b64encode(self.derived_key)
        self.endecrypt = EnDecrypt(self.encrypt_key)
        self.encrypt_keywords = ['password','user']
    
    def iterdict(self, d, mode):
        def iterlist_dict(ld):
            for i in ld:
                self.iterdict(i, mode)

        for k,v in d.items():
            if isinstance(v, dict) :
                self.iterdict(v, mode)
            elif isinstance(v, list):
                iterlist_dict(v)
            else:
                lower_k = k.lower()
                for keyword in self.encrypt_keywords:
                    if keyword in lower_k:
                        if mode == 'encrypt':
                            d[k] = self.endecrypt.encrypt(str(v))
                        elif mode == 'decrypt':
                            d[k] = self.endecrypt.decrypt(str(v))

    def write_encrypt_info(self, input_path, output_path):
        database_info = ''
        with open(f'{input_path}','r') as f:
            database_info = json.loads(f.read())
        
        self.iterdict(database_info,mode='encrypt')

        with open(f'{output_path}','w', encoding='utf-8') as f:
            f.write(json.dumps(database_info,ensure_ascii=False))

    def decrypt_database_info(self, encrypt_db_info_path):
        with open(f'{encrypt_db_info_path}', 'r', encoding='utf-8') as f:
            en_db = json.loads(f.read())
        en_db = dict(en_db)
        self.iterdict(en_db,mode='decrypt')
        return en_db

def get_filter(field_name, filter_condition, filter_value):
    if filter_condition.strip() == "contains":
        kwargs = {
            '{0}__icontains'.format(field_name): filter_value
        }
        return Q(**kwargs)

    if filter_condition.strip() == "not_equal":
        kwargs = {
            '{0}__iexact'.format(field_name): filter_value
        }
        return ~Q(**kwargs)

    if filter_condition.strip() == "starts_with":
        kwargs = {
            '{0}__istartswith'.format(field_name): filter_value
        }
        return Q(**kwargs)
    if filter_condition.strip() == "equal":
        kwargs = {
            '{0}__iexact'.format(field_name): filter_value
        }
        return Q(**kwargs)

    if filter_condition.strip() == "like":
        kwargs = {
            '{0}__like'.format(field_name): filter_value
        }
        return Q(**kwargs)
    if filter_condition.strip() == "in":
        kwargs = {
            '{0}__in'.format(field_name): filter_value
        }
        return Q(**kwargs)

if __name__ == '__main__':
    import sys
    import os
    from pathlib import Path
    APP_DIR = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(APP_DIR))
    os.environ['DJANGO_SETTINGS_MODULE'] = 'config.settings.base'
    os.environ['DJANGO_ALLOW_ASYNC_UNSAFE'] = 'true'
    import django
    django.setup()
    from config.constants import SECRET_FOLDER_PATH
    from django.db.models import Q

    db_info_path = f'{SECRET_FOLDER_PATH}/database_info.json'
    en_db_info_path = f'{SECRET_FOLDER_PATH}/encrypt_database_info.json'
    print(en_db_info_path)
    tde = TempDatabaseInfoEnDecrypt()
    tde.write_encrypt_info(input_path=db_info_path, output_path=en_db_info_path)
    # x = tde.decrypt_database_info(encrypt_db_info_path=en_db_info_path)
    # from django.conf import settings