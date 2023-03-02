## Euclidsoft-Data-Analysis-Platform Structure
#### 작성자 : 권영호 (yhkwon@euclidsoft.co.kr)
#### 수정날짜: 2021.07.27
```
.
├── analysis : 분석 관련 모듈 모음
│   ├── edgelist 
│   │   └── edgelist.py : 엣지리스트 생성 모듈
│   ├── sna
│   │   ├── sna.py : sna 분석 모듈
│   │   ├── sna_functions.py : sna 분석모듈에서 사용 함수 모음
│   │   ├── tree.py : 다차원 sna 분석 모듈
│   │   ├── tree_functions.py : 다처원 sna 분석 모듈 사용 함수 모음
│   │   └── tree_sample.json : 다차원 결과 샘플 
│   ├── urls.py : API URL 주소 설정
│   ├── views.py : 분석 API Http Request, Response 처리
│   ├── word_extractors : 용어 추출 모듈
│   │   ├── noun_extractor.py : 명사/복합어 추출 모듈 (soynlp)
│   │   ├── noun_predict_model : soynlp 에서 사용하는 조사/어미 데이터
│   │   │   ├── noun_predictor_patent_neg : 용언/어미 모음
│   │   │   └── noun_predictor_patent_pos : 조사/형용사 모음
│   │   └── topic_extractor.py : 토픽/연결어/불용어 추출 모듈
│   └── wordcount : 통계/용어 관련 지표 생성 모듈 (작성 예정)
│       └── wordcount.py : keit-rome 에서 사용했던 워드카운터 모듈 (수정 필요)
├── config : django base 분석 플랫폼 전체 설정 관련 모음
│   ├── asgi.py : django 기본 모듈, 파이썬 - Http 연결 관련
│   ├── auth_router.py : django 메타 데이터 관련 데이터베이스 연결 라우터
│   ├── constants.py : 공용으로 사용하는 상수 변수 모음
│   ├── elasticsearch_settings : 엘라스틱서치 관련 데이터
│   │   └── elasticsearch_index_template.py : 엘라스틱서치 인덱스 베이스 템플릿
│   ├── settings
│   │   ├── base.py : 베이스 설정
│   │   ├── dev.py : 개발용 설정
│   │   └── prod.py : 운영용 설정
│   ├── topic_model_router.py : 토픽모델 관련 데이터베이스 연결 라우터
│   ├── urls.py : API URL 주소 1-depth 설정 
│   └── wsgi.py : django 기본 모듈, 파이썬 - Http 연결 관련
├── db_manager : 데이터베이스 관련 관리
│   ├── managers.py : mysql, elasticsearch 관련 데이터베이스 조작 및 관리 모듈
├── django_start.sh : docker 용 django 실행 스크립트
├── documents : 문서 관리
│   ├── models.py : document 데이터베이스 ORM model 
├── manage.py : django 관리 모듈
├── media : 모듈에서 사용하는 리소스 저장소
│   └── userdict : 생성된 사용자 사전 저장
│       ├── docu_patent : 각 document 별로 사전 생성
│       │   ├── representative.txt : 대표어 사전
│       │   ├── stopword.txt : 불용어 사전
│       │   ├── synonym.txt : 동의어 사전
│       │   └── terminology.txt : 명사/복합어 사전
│       ├── docu_subject
│       │   ├── representative.txt
│       │   ├── stopword.txt
│       │   ├── synonym.txt
│       │   └── terminology.txt
│       ├── docu_test
│       │   ├── representative.txt
│       │   ├── stopword.txt
│       │   ├── synonym.txt
│       │   └── terminology.txt
│       └── docu_thesis
│           ├── representative.txt
│           ├── stopword.txt
│           ├── synonym.txt
│           └── terminology.txt
├── requirements : 의존 라이브러리 명세서 (pip install 용)
│   ├── base.txt : 베이스 라이브러리 목록
│   ├── dev.txt : 개발용 라이브러리 목록
│   └── prod.txt : 운영용 라이브러리 목록
├── terms : 용어 관련 모듈
│   ├── models.py : 용어 관련 데이터베이스 ORM 모델
│   ├── serializers.py : 모델 직렬화
│   ├── term_utils.py : 용어 관리용 함수 모음
│   ├── urls.py : 용어 API URL 설정
│   ├── userdict.py : 사용자사전 생성 모듈
│   └── views.py : 용어 API Http Request, Response 처리
├── topic_model : 토픽모델 관련 모듈 (작성 필요)
│   ├── models.py
│   └── views.py
├── utils : 유틸리티 함수 모음
│   ├── common_utils.py : 공통 사용 함수 모음
│   ├── custom_errors.py : 엔진 자체 정의 에러 모음

```