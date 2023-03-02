import os
import paramiko
import pandas as pd
from pathlib import Path
from django.core.files.storage import DefaultStorage
from pathlib import Path

if __name__ != '__main__':
    from db_manager.managers import ElasticManager
    from config.constants import DOCUMENT_DB_NAMES, ELASTICSEARCH_CLUSTER_INFO
    from terms.term_utils import DOCU_TERM_MAPPER
    from terms.userdict_utils import english_only_pattern,word_only_pattern

def get_userdict_file_path(output_path):
    ds = DefaultStorage()
    base_dir = ds.location + '/userdict'
    file = Path(base_dir + output_path)
    file.parent.mkdir(parents=True, exist_ok=True)
    return file


class NoriUserdictMaker():
    def __init__(self,target_db):
        self.target_db = target_db

    def create_noun_userdict(self):
        stopword_obj = DOCU_TERM_MAPPER['stopword']['model'].objects.using(self.target_db).all().values('stopword')
        term_obj = DOCU_TERM_MAPPER['terminology']['model'].objects.using(self.target_db).all().values('term')
        compound_obj = DOCU_TERM_MAPPER['compound']['model'].objects.using(self.target_db).all().values('compound','component')

        stopword_df = pd.DataFrame(stopword_obj)
        term_df = pd.DataFrame(term_obj)
        compound_df = pd.DataFrame(compound_obj)

        if term_df.empty and compound_df.empty:
            return term_df
        stopword_set = set()
        if not stopword_df.empty:
            stopword_set = set(stopword_df.stopword.tolist())
        term_set = set()
        comp_set = set()
        combined_comp_set = set()

        if not term_df.empty:
            term_df['term'] = term_df['term'].map(lambda x: x.replace(' ',''))
            term_set = set(term_df['term'])

        if not compound_df.empty:
            # 전체 component == 불용어 제거
            compound_df.loc[:, 'component_list'] = compound_df.component.map(lambda x: x.split(' '))
            compound_df['comp-stop'] = compound_df.component_list.map(lambda x: len(set(x).difference(stopword_set)))
            compound_df = compound_df.loc[compound_df['comp-stop'] != 0][['compound','component']]
            # 사용자 사전 형태 생성
            compound_df['comp'] = compound_df.apply(' '.join, axis=1)
            combined_comp_set = set(compound_df['comp'])
            comp_set = set(compound_df['compound'])

        term_set = term_set - comp_set
        userdict_list = list(term_set.union(combined_comp_set))
        userdict_df = pd.DataFrame({'term':userdict_list})
        userdict_df = userdict_df[userdict_df.term != '']
        userdict_df = userdict_df.sort_values('term')
        return userdict_df

    def create_synonym_userdict(self):
        synonym_obj = DOCU_TERM_MAPPER['synonym']['model'].objects.using(self.target_db).all().values('main_term','sub_term')
        synonym_df = pd.DataFrame(synonym_obj)

        if synonym_df.empty:
            return synonym_df
        synonym_df.columns = ['main_term','sub_term']
        synonym_df = synonym_df[synonym_df.main_term != synonym_df.sub_term]
        grouped_df = synonym_df.groupby('main_term')
        grouped_df = grouped_df['sub_term'].agg(lambda x: ','.join(x)).reset_index()
        grouped_df['term'] = grouped_df.apply(','.join, axis=1)
        grouped_df = grouped_df.drop_duplicates('term')
        grouped_df = grouped_df.sort_values('term')
        return grouped_df['term'].to_frame()

    def create_representative_userdict(self):
        rep_obj = DOCU_TERM_MAPPER['representative']['model'].objects.using(self.target_db).all().values('main_term','sub_term')
        rep_df = pd.DataFrame(rep_obj)

        if rep_df.empty:
            return rep_df
        rep_df.columns = ['main_term','sub_term']
        rep_df = rep_df[rep_df.main_term != rep_df.sub_term]
        grouped_df = rep_df.groupby('main_term')
        grouped_df = grouped_df['sub_term'].agg(lambda x: ','.join(x)).reset_index()
        grouped_df['term'] = grouped_df.apply(lambda col: col['sub_term']+' => '+col['main_term'],axis=1)
        grouped_df = grouped_df.drop_duplicates('term')
        grouped_df = grouped_df.sort_values('term')
        return grouped_df['term'].to_frame()

    def create_stopword_userdict(self):
        stop_obj = DOCU_TERM_MAPPER['stopword']['model'].objects.using(self.target_db).all().values('stopword')
        stop_df = pd.DataFrame(stop_obj)
        if stop_df.empty:
            return stop_df
        stop_df = stop_df[stop_df.stopword != '']
        stop_df = stop_df.drop_duplicates('stopword')
        stop_df = stop_df.sort_values('stopword')
        return stop_df

def create_nori_tokenizer_userdict(target_db, term_type, output_path=None):
    if not target_db in DOCUMENT_DB_NAMES:
        raise ValueError(f'{target_db} not exists')
    if not output_path:
        output_path = f'/{target_db}/{term_type}.txt'
    file_path = get_userdict_file_path(output_path)
    dict_df = pd.DataFrame()
    nud_maker = NoriUserdictMaker(target_db=target_db)

    if term_type == 'terminology' or term_type == 'compound':
        dict_df = nud_maker.create_noun_userdict()
    elif term_type == 'synonym':
        dict_df = nud_maker.create_synonym_userdict()
    elif term_type == 'representative':
        dict_df = nud_maker.create_representative_userdict()
    elif term_type == 'stopword':
        dict_df = nud_maker.create_stopword_userdict()

    dict_df.to_csv(file_path, index=False, header=False, encoding='utf-8', sep='\n')
    result = {'job':'create userdict txt file','status':'success','detail': {'target_db':target_db,'term_type':term_type, 'term_count':len(dict_df),'created_file_path': str(file_path)}}
    return result

def mkdir_p(sftp, remote_directory):
    if remote_directory == '/':
        # absolute path so change directory to root
        sftp.chdir('/')
        return
    if remote_directory == '':
        # top-level relative directory must exist
        return
    try:
        sftp.chdir(remote_directory) # sub-directory exists
    except IOError:
        dirname, basename = os.path.split(remote_directory.rstrip('/'))
        mkdir_p(sftp, dirname) # make parent directories
        sftp.mkdir(basename) # sub-directory missing, so created it
        sftp.chdir(basename)
        return True

def copy_userdict_to_elasticsearch_cluster(target_db, term_type, input_file_path=None):
    es_cluster_info = ELASTICSEARCH_CLUSTER_INFO[target_db]
    result = []
    if not input_file_path:
        input_file_path = get_userdict_file_path(output_path=f'/{target_db}/{term_type}.txt')

    for cluster_info in es_cluster_info:
        try:
            cli = paramiko.SSHClient()
            cli.set_missing_host_key_policy(paramiko.AutoAddPolicy)
            es_home = cluster_info['ELASTICSEARCH_HOME']
            es_user = cluster_info['SSH_USER']
            es_hostname = cluster_info['SSH_IP']
            es_port = cluster_info['SSH_PORT']
            cluster_password = cluster_info['SSH_PASSWORD']
            sub_result = {'job':'','status':'','detail': {'target_db':target_db,'term_type':term_type, 'remote_hostname': es_hostname}}

            remote_dictionary_dir_path = es_home+f'/config/userdict/{target_db}'
            target_file_name = f'{term_type}.txt'
            remote_file_path = remote_dictionary_dir_path + f'/{target_file_name}'

            cli.connect(hostname=es_hostname, port=es_port, username=es_user, password=cluster_password)
            sftp = cli.open_sftp()
            mkdir_p(sftp=sftp, remote_directory=remote_dictionary_dir_path)
            sftp.put(input_file_path, remote_file_path)
            stdin, stdout, stderr = cli.exec_command(f'ls {remote_dictionary_dir_path} | grep {target_file_name} ')
            lines = stdout.readlines()
            if len(lines) == 1:
                sub_result.update({'job':'copy userdict to remote server','status':'success', 'error': '' })
            else:
                sub_result.update({'job':'copy userdict to remote server','status':'fail', 'error':'need to inspect, check remote server connection'})
            result.append(sub_result)
        except paramiko.ssh_exception.AuthenticationException as e:
            sub_result.update({'job':'copy userdict to remote server','status':'fail', 'error':f'{e}'})
            result.append(sub_result)
            continue
        finally:
            cli.close()
    return result

def create_copy_userdict(target_db, term_type):
    create_result = create_nori_tokenizer_userdict(target_db=target_db, term_type=term_type)
    copy_result = copy_userdict_to_elasticsearch_cluster(target_db=target_db, term_type=term_type)
    return create_result, copy_result


def get_initial_stop_word_list():
    ds = DefaultStorage()
    file_path = ds.location + '/initial_dict/stop_word.parquet'
    base_df = pd.read_parquet(file_path)
    if base_df.empty:
        raise Exception('file contents not exists')
    if not "stopword" in base_df.columns:
        raise Exception('stopword column not exists')
    return base_df.stopword.tolist()


def get_initial_word_list():
    ds = DefaultStorage()
    file_path = ds.location + '/initial_dict/dicterm.parquet'
    base_df = pd.read_parquet(file_path)
    term_list = []
    comp_list = []
    synonym_list = []
    if base_df.empty:
        raise Exception('file contents not exists')

    for col in base_df.columns:
        base_df[col] = base_df[col].map(lambda x: x.lower() if x is not None else x)
        base_df[col] = base_df[col].map(lambda x: None if x =='' else x)
    term_set = set(base_df[base_df.spaced_term.isna()==True]['term'].tolist() +  base_df[base_df.spaced_synonym.isna()==True]['synonym'].tolist())
    comp_set = set(base_df[base_df.spaced_term.isna()==False]['spaced_term'].tolist() + base_df[base_df.spaced_synonym.isna()==False]['spaced_synonym'].tolist())
    if '' in term_set:
        term_set.remove('')
    if '' in comp_set:
        comp_set.remove('')

    comp_df = pd.DataFrame({'component':list(comp_set)})
    if not comp_df.empty:
        comp_df = comp_df.dropna()
        comp_df['compound'] = comp_df.component.map(lambda x: x.replace(' ',''))
        comp_df = comp_df.drop_duplicates('compound').sort_values('compound')
        comp_df['special_mark'] = comp_df.compound.map(word_only_pattern.search)
        comp_df.loc[(comp_df.special_mark.isna()==True),'special_mark'] = comp_df.compound.map(english_only_pattern.search)
        comp_df = comp_df.loc[comp_df.special_mark.isna()]
        comp_df['target_comp'] = comp_df['compound'] + ' ' + comp_df['component']
        comp_list = sorted(comp_df['target_comp'].str.split(' ').tolist())

    term_df = pd.DataFrame({'term':list(term_set)})
    if not term_df.empty:
        term_df = term_df.dropna()
        term_df = term_df.drop_duplicates()
        term_df['special_mark'] = term_df.term.map(word_only_pattern.search)
        term_df.loc[(term_df.special_mark.isna()==True),'special_mark'] = term_df.term.map(english_only_pattern.search)
        term_df = term_df.loc[term_df.special_mark.isna()]
        term_list = term_df['term'].sort_values().tolist()

    synonym_1 = base_df.loc[((base_df.spaced_term.isna())&(base_df.spaced_synonym.isna())),['term','synonym']].rename({'term':'main_term','synonym':'sub_term'},axis=1)
    synonym_2 = base_df.loc[((base_df.spaced_term.isna()==False)&(base_df.spaced_synonym.isna()==False)),['spaced_term','spaced_synonym']].rename({'spaced_term':'main_term','spaced_synonym':'sub_term'},axis=1)
    synonym_3 = base_df.loc[((base_df.spaced_term.isna()==True)&(base_df.spaced_synonym.isna()==False)&(base_df.term.isna()==False)),['term','spaced_synonym']].rename({'term':'main_term','spaced_synonym':'sub_term'},axis=1)
    synonym_4 = base_df.loc[((base_df.spaced_term.isna()==False)&(base_df.spaced_synonym.isna()==True)&(base_df.synonym.isna()==False)),['spaced_term','synonym']].rename({'spaced_term':'main_term','synonym':'sub_term'},axis=1)
    synonym_df = pd.concat([synonym_1,synonym_2,synonym_3,synonym_4],axis=0,ignore_index=True)
    synonym_df = synonym_df.dropna()
    synonym_df = synonym_df.drop_duplicates()
    if not synonym_df.empty:
        synonym_df['special_mark'] = synonym_df.main_term.map(word_only_pattern.search)
        synonym_df.loc[synonym_df.special_mark.isna(),'special_mark'] = synonym_df.sub_term.map(word_only_pattern.search)
        synonym_df = synonym_df[synonym_df.special_mark.isna()]
        synonyms = synonym_df.groupby('main_term')['sub_term'].agg(",".join).reset_index()
        synonyms['synonym'] = synonyms['main_term'] +','+synonyms['sub_term']
        synonyms['synonym_list'] = synonyms['synonym'].map(lambda x: x.split(','))
        synonym_list = synonyms.synonym_list.tolist()
    return term_list, comp_list, synonym_list


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

    from config.constants import DOCUMENT_DB_NAMES, ELASTICSEARCH_CLUSTER_INFO
    from terms.term_utils import DOCU_TERM_MAPPER
    from terms.userdict_utils import english_only_pattern,word_only_pattern
    userdic = NoriUserdictMaker("limenet_analysis")
    userdic.create_noun_userdict()

    # get_initial_word_list()
    # create_nori_tokenizer_userdict(target_db='test_db', term_type='terminology')
    # target_dbs = ['docu_test','docu_patent','docu_thesis','docu_subject']
    # term_types = ['terminology', 'synonym','representative','stopword']
    # for target_db in target_dbs:
    #     for term_type in term_types:
    #         create_nori_tokenizer_userdict(target_db=target_db, term_type=term_type)
    # create_nori_tokenizer_userdict(target_db='docu_test2',term_type='synonym')
    # r = copy_userdict_to_elasticsearch_cluster('docu_test', term_type='synonym')
    # r= get_userdict_file_path(output_path='/docu_test/synonym.txt')