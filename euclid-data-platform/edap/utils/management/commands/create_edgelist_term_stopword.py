from django.core.management.base import BaseCommand, CommandError
from config.constants import DATABASES
from django.core import management
from config.constants import DOCUMENT_DB_NAMES
from analysis.edgelist.kistep_edgelist import SnaEdgelist
from analysis.word_extractors.topic_extractor import make_rawdata_class, make_base_dict, preprocessing_create_edgelist_term
from terms.term_utils import create_edgelist_term_from_word_list, create_stopword_from_word_list
import timeit

class Command(BaseCommand):
    help = 'create and insert recommend term set (edgelist_term, stopword)'

    def add_arguments(self, parser):
        parser.add_argument('--target_db',nargs='+', type=str)

    def handle(self, *args, **options):
        start_time = timeit.default_timer()        
        target_db_list = options['target_db']
        if not options['target_db']:
            target_db_list = DOCUMENT_DB_NAMES
        for _target_db in target_db_list:
            self.stdout.write(self.style.NOTICE(f'start {_target_db} stopword, edgelist_term create'))
            we_edge = SnaEdgelist()
            all_edgelist = we_edge.get_all_edgelist(target_db=_target_db)
            Edgelist_class = make_rawdata_class(all_edgelist)
            Edgelist_class = preprocessing_create_edgelist_term(Edgelist_class)
            middle_all, topic_all, stop_all = make_base_dict(Edgelist_class, qs=0.001, qt=0.2, nm=50)
            self.stdout.write(self.style.NOTICE(f'{_target_db} stopword, edgelist_term create finished'))
            create_stopword_from_word_list(_target_db, stop_all, chunk_size=10000, exclude_stopword=True)
            self.stdout.write(self.style.SUCCESS(f'{_target_db} stopword insert finished'))
            create_edgelist_term_from_word_list(_target_db, topic_all, chunk_size=10000, exclude_stopword=True, insert_terminology=False)
            self.stdout.write(self.style.SUCCESS(f'{_target_db} edgelist_term insert finished'))
        
        self.stdout.write(self.style.SUCCESS(f'create_insert recommended edgelist_term and stopword job finished'))
        self.stdout.write(self.style.NOTICE(f'took: {timeit.default_timer()-start_time} seconds'))