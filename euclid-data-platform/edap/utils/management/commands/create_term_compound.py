from django.core.management.base import BaseCommand, CommandError
from config.constants import DOCUMENT_DB_NAMES
from terms.term_utils import DOCU_TERM_MAPPER
from analysis.word_extractors.topic_extractor import make_rawdata_class, make_base_dict
from terms.term_utils import create_noun_compound_from_noun_extractor, create_noun_compound_from_keyword_col
import timeit

class Command(BaseCommand):
    help = 'create and insert recommend term set (terminology, compound)'

    def add_arguments(self, parser):
        parser.add_argument('--target_db',nargs='+', type=str)

    def handle(self, *args, **options):
        start_time = timeit.default_timer()        
        target_db_list = options['target_db']
        if not options['target_db']:
            target_db_list = DOCUMENT_DB_NAMES
        for _target_db in target_db_list:
            self.stdout.write(self.style.NOTICE(f'start {_target_db} terminology, compound create'))
            noun, compound = create_noun_compound_from_noun_extractor(_target_db, insert_db=True, chunk_size=10000)
            self.stdout.write(self.style.SUCCESS(f'{_target_db} terminology, compound insert finished'))
            kwd_noun, kwd_compound = create_noun_compound_from_keyword_col(_target_db, insert_db=True, chunk_size=10000)
            self.stdout.write(self.style.SUCCESS(f'{_target_db} -keyword- terminology, compound insert finished'))
        self.stdout.write(self.style.SUCCESS(f'create_insert recommended terminology and compound job finished'))
        self.stdout.write(self.style.NOTICE(f'took: {timeit.default_timer()-start_time} seconds'))