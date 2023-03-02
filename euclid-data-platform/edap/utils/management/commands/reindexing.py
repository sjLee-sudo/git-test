from django.core.management.base import BaseCommand, CommandError
from config.constants import DOCUMENT_DB_NAMES
from terms.term_utils import DOCU_TERM_MAPPER
from db_manager.managers import ElasticManager
from analysis.word_extractors.topic_extractor import make_rawdata_class, make_base_dict
from terms.term_utils import create_noun_compound_from_noun_extractor
import timeit
import getpass

class Command(BaseCommand):
    help = 'reindex'

    def add_arguments(self, parser):
        parser.add_argument('--target_db',nargs='+', type=str)

    def handle(self, *args, **options):
        start_time = timeit.default_timer()        
        target_db_list = options['target_db']
        if not options['target_db']:
            target_db_list = DOCUMENT_DB_NAMES
        for _target_db in target_db_list:
            sub_start_time = timeit.default_timer()        
            em = ElasticManager(target_db=_target_db)
            result = em.reindexing(force_migrate=False, create_dictionary=True)
            self.stdout.write(self.style.SUCCESS(f'{_target_db} reindexing finished'))
            self.stdout.write(self.style.SUCCESS(f'{_target_db} reindexing took: {timeit.default_timer()-sub_start_time} seconds'))
        
        self.stdout.write(self.style.SUCCESS(f'total took: {timeit.default_timer()-start_time} seconds'))
        