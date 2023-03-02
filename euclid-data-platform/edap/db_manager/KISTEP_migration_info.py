MIGRATION_INFO_SAMPLE_FORMAT = {
    "MIGRATION_INFO":{
        "PATENT":{
            "MERGE_PROCESS":[
                {
                    # 불러올 데이터베이스 접속 정보 
                    "SOURCE_DB_ACCESS_INFO": "SOURCE_ACCESS_INFO_KEY",
                    # 소스 테이블 이름
                    "SOURCE_TABLE": "SOURCE_TABLE_NAME",
                    # WHERE 조건  (WHERE 빼고 조건만 입력)
                    "SELECT_CONDITION": "COL != VAL1 AND COL2 = VAL2",
                    # WHERE 조건에 특정 컬럼 LIST 에 해당하는 데이터만 불러오기 추가
                    # PK 컬럼값이 존재하는 데이터만 불러오기에 해당 (필요 데이터만 로드) 
                    "FILTER_BY_COL_VALS": ["SBJT_ID"],
                    # 필요한 컬럼명
                    "SOURCE_COLS" : [
                        "*"
                    ],
                    # 전처리 작업 나열
                    "PREPROCESS": [
                        # API 명세서 확인하여 지원하는 데이터 처리 명령어 키워드 입력
                        # 추가적인 상세 설정이 필요한 경우 API 명세서 참고하여 params에 필요한 값 입력.
                        # params 미 입력 시 default 값 적용
                        # 모든 컬럼을 하나씩 순차적으로 동일 job 적용 시 target_cols = ["ALL"]
                        # 전체 DF 에 대해 작업을 진행 필요 시 target_cols = ["SELF"]
                        # remark 에 작업 내용에 대한 주석 정리 가능 (실제 작업에 영향 x)
                        {"job":"job_to_do","target_cols":["target_col1","target_col2"],"params":{"condition1":"val1"},"remark":"comment"},
                    ],
                    # 후처리 작업 나열 (앞선 DF와 JOIN 이후 진행할 작업 나열)
                    "POSTPROCESS":[
                        {"job":"job_to_do","target_cols":["target_col1","target_col2"],"params":{"condition1":"val1"}},
                    ],
                    # 앞에서 작업한 DF 와 JOIN 필요 시 JOIN 조건으로 사용할 키 컬럼
                    "JOIN_KEY": ["SBJT_ID"],
                    # TARGET_DB_ACCESS_INFO 입력 시 DF 를 테이블 OR PARQUET 파일 OR IN MEMORY로 저장
                    # 해당 키값이 없거나 빈값 기입 시 따로 저장 작업 진행하지 않음
                    # DB 저장 시 ACCESS_INFO 의 키값, parquet 저장 시 "PARQUET"으로 기입, IN MEMORY로 보관 시 "IN_MEMORY"로 기입
                    "TARGET_DB_ACCESS_INFO" : "INSERT_ACCESS_INFO_KEY",
                    # DB 저장 시 테이블명, parquet 저장 시 저장 경로 및 파일이름
                    "TARGET_TABLE": "PS_SBJT",
                    # 코멘트 정보가 있으면 코멘트 정보 저장
                    "ADD_COMMENT": "Y",
                    # 주석
                    "REMARK":"입력값 설명 포맷"
                }
            ]
        }
    }
}
ACCESS_INFO = {
    "ORIGINAL_SUBJECT":{
        "NAME":"ORCLCDB",
        "ENGINE":"oracle+cx_oracle",
        "USER": "C##TEST_USER2",
        "PASSWORD": "1234",
        "HOST": "192.168.1.53",
        "PORT": 1521
    },
    "COPY_SUBJECT":{
        "NAME":"docu_copy",
        "ENGINE":"mysql+pymysql",
        "USER": "airflow_user",
        "PASSWORD": "airflow_pass",
        "HOST": "192.168.1.53",
        "PORT": 43306
    },
    "ORIGINAL_STANDARD_MANAGEMENT":{
        "NAME":"ORCLCDB",
        "ENGINE":"oracle+cx_oracle",
        "USER": "C##TEST_USER3",
        "PASSWORD": "1234",
        "HOST": "192.168.1.53",
        "PORT": 1521
    },
    "COPY_STANDARD_MANAGEMENT":{
        "NAME":"docu_test",
        "ENGINE":"mysql+pymysql",
        "USER": "airflow_user",
        "PASSWORD": "airflow_pass",
        "HOST": "192.168.1.53",
        "PORT": 43306
    },
    "ANALYSIS_SUBJECT":{
        "NAME":"docu_test",
        "ENGINE":"mysql+pymysql",
        "USER": "airflow_user",
        "PASSWORD": "airflow_pass",
        "HOST": "192.168.1.53",
        "PORT": 43306
    }
}
DEFAULT_SOURCE = "ORIGINAL_SUBJECT"
DEFAULT_TARGET = "ANALYSIS_SUBJECT"
COPY_SOURCE = "COPY_SUBJECT"

ORIGINAL_STANDARD_MANAGEMENT = "ORIGINAL_STANDARD_MANAGEMENT"
COPY_STANDARD_MANAGEMENT = "COPY_STANDARD_MANAGEMENT"

MIGRATION_INFO = {    
    "ORIGINAL_COPY":{
        "MERGE_PROCESS": [
            {
                "SOURCE_DB_ACCESS_INFO": DEFAULT_SOURCE,
                "SOURCE_TABLE": "PS_SBJT",
                "TARGET_DB_ACCESS_INFO" : DEFAULT_TARGET,
                "TARGET_TABLE": "PS_SBJT",
                "SOURCE_COLS" : [
                    "*"
                ],
                "ADD_COMMENT": "Y",
                "REMARK":"연구개발 과제 테이블, 과제 ID 및 기본 정보 추출용 테이블"
            },
            {
                "SOURCE_DB_ACCESS_INFO": DEFAULT_SOURCE,
                "SOURCE_TABLE": "PS_SBJT_KWD",
                "TARGET_DB_ACCESS_INFO" : DEFAULT_TARGET,
                "TARGET_TABLE": "PS_SBJT_KWD",
                "SOURCE_COLS" : [
                    "*"
                ],
                "ADD_COMMENT": "Y",
                "REMARK":"과제 키워드"
            },
            {
                "SOURCE_DB_ACCESS_INFO": DEFAULT_SOURCE,
                "SOURCE_TABLE": "PS_SBJT_TOT_ANNL",
                "TARGET_DB_ACCESS_INFO" : DEFAULT_TARGET,
                "TARGET_TABLE": "PS_SBJT_TOT_ANNL",
                "SOURCE_COLS" : [
                    "*"
                ],
                "ADD_COMMENT": "Y",
                "REMARK":"과제 목표, 범위 내용, 기대효과 sbjt_id 로 합침"
            },
            {
                "SOURCE_DB_ACCESS_INFO": DEFAULT_SOURCE,
                "SOURCE_TABLE": "PS_CFRSR_IPRS",
                "TARGET_DB_ACCESS_INFO" : DEFAULT_TARGET,
                "TARGET_TABLE": "PS_CFRSR_IPRS",
                "SOURCE_COLS" : [
                    "*"
                ],
                "ADD_COMMENT": "Y",
                "REMARK":"과제 연구책임자 지식재산권 테이블, 국내외특허 출원 구분 포함"
            },
            {
                "SOURCE_DB_ACCESS_INFO": DEFAULT_SOURCE,
                "SOURCE_TABLE": "PS_SBJT_ORGN_RSCH_EXE",
                "TARGET_DB_ACCESS_INFO" : DEFAULT_TARGET,
                "TARGET_TABLE": "PS_SBJT_ORGN_RSCH_EXE",
                "SOURCE_COLS" : [
                    "*"
                ],
                "ADD_COMMENT": "Y",
                "REMARK":"과제 수행기관 연구 수행, 과제 현황 테이블"
            },
            {
                "SOURCE_DB_ACCESS_INFO": DEFAULT_SOURCE,
                "SOURCE_TABLE": "PS_CFRSR_REXE",
                "TARGET_DB_ACCESS_INFO" : DEFAULT_TARGET,
                "TARGET_TABLE": "PS_CFRSR_REXE",
                "SOURCE_COLS" : [
                    "*"
                ],
                "ADD_COMMENT": "Y",
                "REMARK":"과제 연구책임자 연구 수행, 과제 현황 테이블"
            },
            {
                "SOURCE_DB_ACCESS_INFO": DEFAULT_SOURCE,
                "SOURCE_TABLE": "PS_SBJT_ORGN_IPRS",
                "TARGET_DB_ACCESS_INFO" : DEFAULT_TARGET,
                "TARGET_TABLE": "PS_SBJT_ORGN_IPRS",
                "SOURCE_COLS" : [
                    "*"
                ],
                "ADD_COMMENT": "Y",
                "REMARK":"과제 수행기관 지식재산권 테이블, 국내외 출원 구분 포함"
            },
            {
                "SOURCE_DB_ACCESS_INFO": DEFAULT_SOURCE,
                "SOURCE_TABLE": "PS_CFRSR_THES",
                "TARGET_DB_ACCESS_INFO" : DEFAULT_TARGET,
                "TARGET_TABLE": "PS_CFRSR_THES",
                "SOURCE_COLS" : [
                    "*"
                ],
                "ADD_COMMENT": "Y",
                "REMARK":"과제 연구책임자 논문 현황 관리, 주관기관 책임자와 공동연구기관 책임자 해당"
            },
            {
                "SOURCE_DB_ACCESS_INFO": DEFAULT_SOURCE,
                "SOURCE_TABLE": "PS_CFRSR_ETC",
                "TARGET_DB_ACCESS_INFO" : DEFAULT_TARGET,
                "TARGET_TABLE": "PS_CFRSR_ETC",
                "SOURCE_COLS" : [
                    "*"
                ],
                "ADD_COMMENT": "Y",
                "REMARK":"과제 연구책임자 기타 실적"
            },
            {
                "SOURCE_DB_ACCESS_INFO": DEFAULT_SOURCE,
                "SOURCE_TABLE": "PS_SBJT_RSCH_MBR",
                "TARGET_DB_ACCESS_INFO" : DEFAULT_TARGET,
                "TARGET_TABLE": "PS_SBJT_RSCH_MBR",
                "SOURCE_COLS" : [
                    "*"
                ],
                "ADD_COMMENT": "Y",
                "REMARK":"과제를 수행하는 참여기관의 책임자와 참여연구원만 해당"
            },
            {
                "SOURCE_DB_ACCESS_INFO": DEFAULT_SOURCE,
                "SOURCE_TABLE": "PS_SBJT_RSCH_ORGN",
                "TARGET_DB_ACCESS_INFO" : DEFAULT_TARGET,
                "TARGET_TABLE": "PS_SBJT_RSCH_ORGN",
                "SOURCE_COLS" : [
                    "*"
                ],
                "ADD_COMMENT": "Y",
                "REMARK":"과제를 수행하는 참여기관의 책임자와 참여연구원만 해당"
            },
            {
                "SOURCE_DB_ACCESS_INFO": DEFAULT_SOURCE,
                "SOURCE_TABLE": "PS_SBJT_TECL",
                "TARGET_DB_ACCESS_INFO" : DEFAULT_TARGET,
                "TARGET_TABLE": "PS_SBJT_TECL",
                "SOURCE_COLS" : [
                    "*"
                ],
                "ADD_COMMENT": "Y",
                "REMARK":"과제를 수행하는 참여기관의 책임자와 참여연구원만 해당"
            }
        ]

    },
    "SUBJECT": {
        "MERGE_PROCESS": [
            {
                "SOURCE_DB_ACCESS_INFO" : COPY_SOURCE,
                "SOURCE_TABLE": "PS_SBJT",
                "SELECT_CONDITION": "SBJT_PRG_SE = '진행'",
                "SOURCE_COLS" : [
                    "SBJT_ID","HAN_SBJT_NM","ENG_SBJT_NM","BSNS_YY","SORGN_BSNS_CD","BSNS_ANCM_SN","BSNS_PTC_ANCM_SN",
                    "SBJT_PROPL_STRC_SE","PRG_SORGN_BSNS_CD","OVRL_SBJT_ID","OVRS_NOPEN_YN","SBJT_DVLM_TP_SE"
                ],
                "PREPROCESS": [
                    {"job":"drop_na", "target_cols": ["ALL,SELF, COLS"]},
                    {"job":"drop_duplicates","target_cols":["SBJT_ID"]}
                ],
                "POSTPROCESS":[
                    {"":""}
                ],
                "ADD_COMMENT": "Y",
                "REMARK": """
                    연구개발 과제 테이블, 과제 ID 및 기본 정보 추출용 테이블, 작업의 베이스 테이블. 
                    해당 테이블을 기준으로 다른 추가 데이터를 조인해서 사용
                """
            },
            {
                "SOURCE_DB_ACCESS_INFO" : COPY_SOURCE,
                "SOURCE_TABLE": "PS_SBJT_KWD",
                "SOURCE_COLS" : ["SBJT_ID","KWD_SE","KWD_NM","KREN_SE","KWD_OR"],
                "JOIN_KEY": ["SBJT_ID"],
                "SELECT_CONDITION": "",
                "FILTER_BY_COL_VALS": ["SBJT_ID"],
                "PREPROCESS":[
                    {"job":"fillna","target_cols":["ALL"]},
                    {"job":"concat","target_cols":["ALL"],"params":{"on":"SBJT_ID","sep":",","지정안하면 default 값, 아는 사람만 사용 할 수 있게":""}}
                ],
                "REMARK":"과제 키워드"
            },
            {
                "SOURCE_DB_ACCESS_INFO" : COPY_SOURCE,
                "SOURCE_TABLE": "PS_SBJT_TOT_ANNL",
                "SOURCE_COLS" : ["SBJT_ID","RSCH_GOLE_CN","RSCH_RANG_CN","EXPE_EFCT_CN"],
                "SELECT_CONDITION": "",
                "FILTER_BY_COL_VALS": ["SBJT_ID"],
                "JOIN_KEY":["SBJT_ID"],
                "PREPROCESS": [
                    {"job":"","target_cols":["SBJT_ID","RSCH_GOLE_CN"],"remark":""},
                    {"job":"drop_duplicates","target_cols":["SBJT_ID","RSCH_RANG_CN"]},
                    {"job":"drop_duplicates","target_cols":["SBJT_ID","EXPE_EFCT_CN"]},
                    {"job":"concat","target_cols":["RSCH_GOLE_CN"]},
                    {"job":"concat","target_cols":["RSCH_RANG_CN"]},
                    {"job":"concat","target_cols":["EXPE_EFCT_CN"]},
                    {"job":"col_concat","target_cols":["RSCH_GOLE_CN","RSCH_RANG_CN","EXPE_EFCT_CN"],"params":{"col_name":"contents"}}
                ],
                    "REMARK":"과제 종합 연차: 과제 목표, 범위 내용, 기대효과 sbjt_id 로 합침"
            },
            {
                "SOURCE_DB_ACCESS_INFO" : COPY_SOURCE,
                "SOURCE_TABLE": "PS_SBJT_RSCH_ORGN",
                "SOURCE_COLS" : ["SBJT_ID","RSCH_ORGN_NM","RSCH_ORGN_ID","AGRT_ORGN_ID"],
                "SELECT_CONDITION": "RSCH_ORGN_ROLE_SE = '추후확인 필요'",
                "FILTER_BY_COL_VALS": ["SBJT_ID"],
                "JOIN_KEY":["SBJT_ID"],
                "PREPROCESS": [
                    {"job":"adapter_cleaning_organ_nm","target_cols":["RSCH_ORGN_NM"]},
                    {"job":"concat","target_cols":["RSCH_ORGN_NM"]},
                    {"job":"concat","target_cols":["RSCH_ORGN_ID"]},
                    {"job":"concat","target_cols":["AGRT_ORGN_ID"]}
                ],
                "ADD_COMMENT": "Y",
                "REMARK":"연구를 직접 수행하는 주관기관 + 공동연구기관 + 수요처기관의 기본정보 관리"
            },
            {
                "SOURCE_DB_ACCESS_INFO" : COPY_SOURCE,
                "SOURCE_TABLE": "PS_SBJT_RSCH_MBR",
                "SOURCE_COLS" : [
                    "SBJT_ID","RSCH_ORGN_ID","RSCR_MBR_ID","AGRT_ORGN_ID","SCH_NM","PSTN_NM","DEG_SE",
                    "RSCH_CHRG_FILD_CN", "RSCR_NM"
                ],
                "SELECT_CONDITION": "RSCH_ORGN_ROLE_SE = '추후확인 필요'",
                "FILTER_BY_COL_VALS": ["SBJT_ID"],
                "JOIN_KEY":["SBJT_ID"],
                "PREPROCESS": [
                    {"job":"adapter_cleaning_organ_nm","target_cols":["RSCH_ORGN_NM"]},
                    {"job":"concat","target_cols":["RSCH_ORGN_NM"]},
                    {"job":"concat","target_cols":["RSCH_ORGN_ID"]},
                    {"job":"concat","target_cols":["AGRT_ORGN_ID"]},
                    # {"job":"df_to_dict","target_cols":["SELF"],"params":{"index":["SBJT_ID"],"col_name":"PS_SBJT_RSCH_MBR"}}
                    {"job":"df_to_dict","params":{"index":["SBJT_ID"],"col_name":"PS_SBJT_RSCH_MBR"}}
                ],
                "ADD_COMMENT": "Y",
                "REMARK":"과제를 수행하는 참여기관의 책임자와 참여연구원만 해당"
            },
            {
                "SOURCE_DB_ACCESS_INFO" : COPY_SOURCE,
                "SOURCE_TABLE": "PS_SBJT_TECL",
                "SOURCE_COLS" : [
                    "SBJT_ID","SBJT_TECL_CD","PRIO_RK","WGHT_PT"
                ],
                "SELECT_CONDITION": "",
                "FILTER_BY_COL_VALS": ["SBJT_ID"],
                "JOIN_KEY":["SBJT_ID"],
                "PREPROCESS": [
                    {"job":"sort","target_cols":["WGHT_PT"],"params":{"order_by":"desc"},"remark":"분류 코드 가중치 높은 순으로 정렬"},
                    {"job":"drop_duplicates","target_cols":["SBJT_ID"],"params":{"left":"first"},"remark":"가중치 낮은 SBJT_ID 중복데이터 제거"}
                ],
                "ADD_COMMENT": "Y",
                "REMARK":"과제에 대한 다양한 기술적 + 일반 분류를 통합관리"
            },
            {
                "TARGET_DB_ACCESS_INFO" : "DEFAULT_TARGET2",
                "TARGET_TABLE": "target_table_name",
                "PREPROCESS": [
                    {}
                ],
                "POSTPROCESS":[
                    {"job":"rename_cols","target_cols":[{"SBJT_ID":"DOC_ID","RSCH_ORGN_ID":"PLAYER_ID"}]}
                ]
            }                
        ]
    },
    "PATENT":{
        "MERGE_PROCESS": [
            {
                "SOURCE_DB_ACCESS_INFO" : "PATENT_COPY",
                "SOURCE_TABLE": "PS_SBJT_ORGN_IPRS",
                "SELECT_CONDITION": "IPRS_SE = '특허'",
                "SOURCE_COLS" : ["SBJT_ID","IPRS_SE","IPRS_NM","RSCH_ORGN_ID","APLYR_NM_LST","INVTR_NM_LST","IPRS_ABST_CN","APLY_NO","APLY_DE"],
                "PREPROCESS": [
                    {}
                ],
                "TARGET_DB_ACCESS_INFO":"IN_MEMORY",
                "TARGET_TABLE":"patent_df",
                "POSTPROCESS":[
                    {
                        "job":"rename_col", 
                        "target_cols":[
                            {
                                "APLY_NO" : "DOC_ID",
                                "IPRS_NM":"TITLE",
                                "APLY_REG_DE": "APP_DT",
                                "APLYR_NM_LST": "PUBLISHER",
                                "INVTR_NM_LST":"INVENTORS",
                            }
                        ]
                    }
                ],
                "REMARK":""
            }
        ]            
    }
}