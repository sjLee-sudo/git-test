from django.core.management.base import BaseCommand
from config.constants import DOCUMENT_DB_NAMES
from django.core.files.storage import DefaultStorage
from terms.term_utils import create_compound_from_word_list, create_edgelist_term_from_word_list, create_synonym_and_representative_from_word_list, create_terminology_from_word_list, create_stopword_from_word_list
from terms.userdict import get_initial_word_list, get_initial_stop_word_list
import pandas as pd
import timeit
import re

class Command(BaseCommand):
    help = 'insert initial term, compound, synonym'

    def add_arguments(self, parser):
        parser.add_argument('--target_db',nargs='+', type=str)

    def handle(self, *args, **options):
        start_time = timeit.default_timer()        
        target_db_list = options['target_db']
        if not options['target_db']:
            target_db_list = DOCUMENT_DB_NAMES
        for _target_db in target_db_list:
            self.stdout.write(self.style.NOTICE(f'start {_target_db} initial dict create'))
            
            stopword_list = get_initial_stop_word_list()
            stopword_res = create_stopword_from_word_list(target_db=_target_db, word_list=stopword_list, exclude_stopword=True)
            self.stdout.write(self.style.SUCCESS(f'{_target_db} stopword {len(stopword_res)} rows insert finished'))

            term_list, comp_list, synonym_list = get_initial_word_list()

            term_res = create_terminology_from_word_list(target_db=_target_db,word_list=term_list,exclude_stopword=True)
            self.stdout.write(self.style.SUCCESS(f'{_target_db} terminology {len(term_res)} rows insert finished'))

            edgelist_res = create_edgelist_term_from_word_list(target_db=_target_db,word_list=term_list,exclude_stopword=True,insert_terminology=False)
            self.stdout.write(self.style.SUCCESS(f'{_target_db} edgelist term {len(edgelist_res)} rows insert finished'))

            comp_res = create_compound_from_word_list(target_db=_target_db, compound_list=comp_list)
            self.stdout.write(self.style.SUCCESS(f'{_target_db} compound {len(comp_res)} rows insert finished'))

            synonym_res = create_synonym_and_representative_from_word_list(target_db=_target_db,main_sub_term_list=synonym_list,term_type='synonym')
            self.stdout.write(self.style.SUCCESS(f'{_target_db} synonym {len(synonym_res)} rows insert finished'))


        self.stdout.write(self.style.SUCCESS(f'create_insert initial dict job finished'))
        self.stdout.write(self.style.NOTICE(f'took: {timeit.default_timer()-start_time} seconds'))