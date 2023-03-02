from django.core.management.base import BaseCommand
from config.constants import DOCUMENT_DB_NAMES
from db_manager.managers import DocumentMigrator
import timeit

class Command(BaseCommand):
    help = 'docu data migration'

    def add_arguments(self, parser):
        parser.add_argument('--target_db',nargs='+', type=str)

    def handle(self, *args, **options):
        start_time = timeit.default_timer()        
        target_db_list = options['target_db']
        if not options['target_db']:
            target_db_list = DOCUMENT_DB_NAMES
        for _target_db in target_db_list:
            sub_start_time = timeit.default_timer()
            dm = DocumentMigrator(multiprocess_mode=False)
            result = dm.execute_migrate(target_db=_target_db)
            self.stdout.write(self.style.SUCCESS(f'{_target_db} reindexing finished'))
            self.stdout.write(self.style.SUCCESS(f'{_target_db} reindexing took: {timeit.default_timer()-sub_start_time} seconds'))
        
        self.stdout.write(self.style.SUCCESS(f'total took: {timeit.default_timer()-start_time} seconds'))
        