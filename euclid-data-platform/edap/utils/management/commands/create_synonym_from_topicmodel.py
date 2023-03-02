from django.core.management.base import BaseCommand, CommandError
from config.constants import DATABASES
from django.core import management
from config.constants import DOCUMENT_DB_NAMES
from terms.term_utils import create_topic_synonym_list_from_topicmodel
import timeit


class Command(BaseCommand):
    help = 'create and insert recommend term set (synonym)'

    def add_arguments(self, parser):
        parser.add_argument('--target_db', nargs='+', type=str)

    def handle(self, *args, **options):
        start_time = timeit.default_timer()
        target_db_list = options['target_db']
        if not options['target_db']:
            target_db_list = DOCUMENT_DB_NAMES
        for _target_db in target_db_list:
            self.stdout.write(self.style.NOTICE(f'start {_target_db} -topic- synonym create'))

            create_topic_synonym_list_from_topicmodel(target_db=_target_db,term_type='synonym', insert_db=True)
            self.stdout.write(self.style.SUCCESS(f'{_target_db} synonym insert finished'))

        self.stdout.write(self.style.SUCCESS(f'create_insert recommended -topic- synonym job finished'))
        self.stdout.write(self.style.NOTICE(f'took: {timeit.default_timer() - start_time} seconds'))