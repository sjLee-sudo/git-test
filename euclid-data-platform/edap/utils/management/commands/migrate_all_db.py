from django.core.management.base import BaseCommand, CommandError
from config.constants import DATABASES
from django.core import management
from django.db.utils import OperationalError

class Command(BaseCommand):
    help = 'django migration for all databases'
 
    def handle(self, *args, **options):
        management.call_command('makemigrations')
        for _db, val in DATABASES.items():
            try:
                if val['HOST'] != '':
                    if _db == 'auth_db':
                        for _app_label in ['auth', 'contenttypes','admin','sessions']:
                            management.call_command('migrate', app_label=_app_label, database=_db)
                    else:
                        management.call_command('migrate', database=_db)
                    self.stdout.write(self.style.SUCCESS(f'{_db} django migrate finished'))
            except OperationalError as oe:
                self.stderr.write(f'{oe}')
                continue