CREATE TABLE LIMENET_ANAL.DOCUMENT
(
/*GRND_PS.PS_SBJT_PRG_PTC */
SBJT_ID VARCHAR(8) NOT NULL COMMENT '과제 ID' 
, PRG_SN INT  DEFAULT 1 NOT NULL COMMENT '진행 순번' 
, PTC_PRG_SN INT DEFAULT 1 NOT NULL COMMENT '세부 진행 순번' 
/*GRND_PS.PS_SBJT : SBJT_ID*/
, RND_SBJT_NO VARCHAR(16) NOT NULL COMMENT '연구개발 과제 번호' 
, HAN_SBJT_NM VARCHAR(300) NOT NULL COMMENT '한글 과제 명' 
, ENG_SBJT_NM VARCHAR(300) NOT NULL COMMENT '영문 과제 명' 
, OVRL_SBJT_ID VARCHAR(8) COMMENT '총괄 과제 ID' 
, HAN_OVRL_RND_NM VARCHAR(300) COMMENT '한글 총괄 연구개발 명' 
, ENG_OVRL_RND_NM VARCHAR(300) COMMENT '영문 총괄 연구개발 명' 
, BSNS_YY VARCHAR(4) NOT NULL COMMENT '사업 년도' 
, SORGN_BSNS_CD VARCHAR(7) NOT NULL COMMENT '전문기관 사업 코드' 
    /*총괄/세부/9일반*/
, SBJT_PROPL_STRC_SE VARCHAR(6) NOT NULL COMMENT '과제 추진 체계 구분[AD4]' 
    /*PO_COMM_CD JOIN*/
, SBJT_PROPL_STRC_SE_NM VARCHAR(6) NOT NULL COMMENT '과제 추진 체계 구분명'
    /* 단계/일반*/
, RSCH_PRID_CPST_SE VARCHAR(6) NOT NULL COMMENT '연구 기간 구성 구분[AB5]' 
    /*PO_COMM_CD JOIN*/
, RSCH_PRID_CPST_SE_NM VARCHAR(6) NOT NULL COMMENT '연구 기간 구성 구분명' 
    /*개념계획서,과제조정,신청용 연구개발계획서,차단계 연구개발계획서,협약변경,협약용 연구개발계획서*/
, RCVE_PLDOC_TP_SE VARCHAR(6) NOT NULL COMMENT '접수 계획서 유형 구분[AS5]' 
    /*PO_COMM_CD JOIN*/
, RCVE_PLDOC_TP_SE_NM VARCHAR(6) NOT NULL COMMENT '접수 계획서 유형 구분명' 
    /*원천기술형/혁신제품형/해당사항없음*/
, SBJT_DVLM_TP_SE VARCHAR(6) NOT NULL COMMENT '과제 개발 유형 구분[AA7]' 
    /*PO_COMM_CD JOIN*/
, SBJT_DVLM_TP_SE_NM VARCHAR(6) NOT NULL COMMENT '과제 개발 유형 구분명' 
, OVRS_NOPEN_YN VARCHAR(1) DEFAULT 'Y' NOT NULL COMMENT '대외 비공개 여부' 
, PRG_SBJT_EXE_ANNL TINYINT DEFAULT 1 NOT NULL COMMENT '진행 과제 수행 연차' 
, PRG_BSNS_YY VARCHAR(4) NOT NULL COMMENT '진행 사업 년도' 
, PRG_SORGN_BSNS_CD VARCHAR(7) NOT NULL COMMENT '진행 전문기관 사업 코드' 
    /*개발종(완)료,개발중,계속,과제접수결과,선정,선정평가결과,신청/접수,연구개발계획서제출 대상,중단,탈락,협약체결결과,협약포기*/
, SBJT_PRG_SE VARCHAR(6) NOT NULL COMMENT '과제 진행 구분[AD3]' 
    /*PO_COMM_CD JOIN*/
, SBJT_PRG_SE_NM VARCHAR(6) NOT NULL COMMENT '과제 진행 구분명' 
    /*정상과제,모의과제*/
, SBJT_CRT_SE VARCHAR(6) NOT NULL COMMENT '과제 생성 구분[BP5]' 
    /*PO_COMM_CD JOIN*/
, SBJT_CRT_SE_NM VARCHAR(6) NOT NULL COMMENT '과제 생성 구분명' 
    /*계획서 작성중,기관담당자 제출반려,기관담당자 제출승인,접수마감,접수반려,접수취소,제출완료,제출유예*/
, LAST_RCVE_STT_SE VARCHAR(6) NOT NULL COMMENT '최종 접수 상태 구분[AO4]' 
    /*PO_COMM_CD JOIN*/
, LAST_RCVE_STT_SE_NM VARCHAR(6) NOT NULL COMMENT '최종 접수 상태 구분명' 
/*GRND_PS.PS_SBJT_TOT_ANNL : SBJT_ID, PRG_SN, PTC_PRG_SN*/
, RSCH_STR_DE DATETIME NOT NULL COMMENT '연구 시작 일자' 
, RSCH_END_DE DATETIME NOT NULL COMMENT '연구 종료 일자' 
, RSCH_GOLE_CN TEXT NOT NULL COMMENT '연구 목표 내용' 
, RSCH_RANG_CN TEXT NOT NULL COMMENT '연구 범위 내용' 
, EXPE_EFCT_CN TEXT NOT NULL COMMENT '기대 효과 내용' 
, PRSPT_FRUT_CN TEXT NOT NULL COMMENT '예상 성과 내용' 
/*GRND_PS.PS_SBJT_KWD : SBJT_ID, PRG_SN, PTC_PRG_SN*/
    /* 한/영 구분없이 키워드 순번대로 , 로 묶음*/
, KWD VARCHAR(1000) DEFAULT '' NOT NULL COMMENT '키워드 명' 
/*GRND_PS.PS_SORGN_YY_BSNS : SORGN_BSNS_CD*/
, HIRK_SORGN_BSNS_CD VARCHAR(7) COMMENT '상위 전문기관 사업 코드' 
, SORGN_ID VARCHAR(5) NOT NULL COMMENT '전문기관 ID' 
, SORGN_BSNS_NM VARCHAR(100) NOT NULL COMMENT '전문기관 사업 명' 
    /*내역사업,단위사업,세부사업,프로그램*/
, BSNS_CSRT_SE VARCHAR(6) NOT NULL COMMENT '사업 구조 구분[AH1]'
    /*PO_COMM_CD JOIN */
, BSNS_CSRT_SE_NM VARCHAR(6) NOT NULL COMMENT '사업 구조 구분명' 
, BSNS_TP_SE VARCHAR(6) COMMENT '사업 유형 구분[AH6]' 
    /*PO_COMM_CD JOIN*/
, BSNS_TP_SE_NM VARCHAR(6) COMMENT '사업 유형 구분명' 
, SBJT_SPRT_BUD_AM BIGINT DEFAULT 0 NOT NULL COMMENT '과제 지원 예산 금액' 
, PLNN_EVAL_BUD_AM BIGINT DEFAULT 0 NOT NULL COMMENT '기획 평가 예산 금액' 
, DLGT_ECTN_BSNS_YN VARCHAR(1) DEFAULT 'N' NOT NULL COMMENT '위임 예외 사업 여부' 
    /*계속,분할계속,신규,신규,일반계속,종료,종료,통합계속*/
, BSNS_PRG_SE VARCHAR(6) COMMENT '사업 진행 구분[AH7]' 
    /*PO_COMM_CD JOIN*/
, BSNS_PRG_SE_NM VARCHAR(6) COMMENT '사업 진행 구분명' 
/*GRND_PS.PS_SORGN  : SORGN_ID*/
, HIRK_SORGN_ID VARCHAR(5) COMMENT '상위 전문기관 ID' 
    /*PS_SORGN JOIN*/
, HIRK_SORGN_NM VARCHAR(5) COMMENT '상위 전문기관명' 
, SORGN_NM VARCHAR(300) NOT NULL COMMENT '전문기관 명' 
    /*범(다)부처사업단,전문기관,전문기관 소속 사업단*/
, SORGN_ROLE_SE VARCHAR(6) NOT NULL COMMENT '전문기관 역할 구분[AS3]' 
    /*PO_COMM_CD JOIN*/
, SORGN_ROLE_SE_NM VARCHAR(6) NOT NULL COMMENT '전문기관 역할 구분명' 
    /*경찰청,고용노동부,공정거래위원회,과학기술부,과학기술정보통신부,교육과학기술부,교육부,국무조정실및국무총리비서실,국무총리실,국방부,국토교통부,기상청,기획재정부,노동부,농림축산식품부,농촌진흥청,문화관광부,문화재청,문화체육관광부,미래창조과학부,방송통신위원회,방위사업청,범부처,법무부,법제처,보건복지가족부,보건복지부,산림청,산업통상자원부,새만금개발청,소방청,식품의약품안전처,여성가족부,여성부,외교부,원자력안전위원회,인사혁신처,정보통신부,중소기업청,중소벤처기업부,지식경제부,통일부,특허청,해양경찰청,해양수산부,행정안전부,행정중심복합도시건설청,환경부*/
, BLNG_GOVD_SE VARCHAR(6) NOT NULL COMMENT '소속 부처 구분[AR4]' 
    /*PO_COMM_CD JOIN*/
, BLNG_GOVD_SE_NM VARCHAR(6) NOT NULL COMMENT '소속 부처 구분명' 
, ORGN_ID VARCHAR(8) NOT NULL COMMENT '기관 ID' 
/*GRND_PS.PS_SBJT_RSCH_ORGN : SBJT_ID, PRG_SN, PTC_PRG_SN*/
    /* Table -> json*/
    /*
        , SBJT_EXE_ANNL TINYINT DEFAULT 1 NOT NULL COMMENT '과제 수행 연차' 
        , RSCH_ORGN_ID VARCHAR(8) NOT NULL COMMENT '연구 기관 ID' 
        , RSCH_ORGN_ROLE_SE VARCHAR(6) NOT NULL COMMENT '연구 기관 역할 구분[AB2]' 
            -- PO_COMM_CD JOIN
        , RSCH_ORGN_ROLE_SE_NM VARCHAR(6) NOT NULL COMMENT '연구 기관 역할 구분명' 
        , AGRT_ORGN_ID VARCHAR(8) NOT NULL COMMENT '협약 기관 ID' 
        , RSCH_ORGN_NM VARCHAR(300) NOT NULL COMMENT '연구 기관 명' 
        , ORGN_STP_PURS_SE VARCHAR(6) NOT NULL COMMENT '기관 설립 목적 구분[PL9]' 
            -- PO_COMM_CD JOIN
        , ORGN_STP_PURS_SE_NM VARCHAR(6) NOT NULL COMMENT '기관 설립 목적 구분명' 
        , ORGN_TP_SE VARCHAR(6) NOT NULL COMMENT '기관 유형 구분[PL7]' 
            -- PO_COMM_CD JOIN
        , ORGN_TP_SE VARCHAR(6) NOT NULL COMMENT '기관 유형 구분명' 
        , NAT_SE VARCHAR(6) NOT NULL COMMENT '국가 구분[PH1]' 
            -- PO_COMM_CD JOIN
        , NAT_SE_NM VARCHAR(6) NOT NULL COMMENT '국가 구분명' 
        , LOC_ZNE_SE VARCHAR(6) NOT NULL COMMENT '소재지 지역 구분[PH8]' 
            -- PO_COMM_CD JOIN
        , LOC_ZNE_SE_NM VARCHAR(6) NOT NULL COMMENT '소재지 지역 구분명' 
        , CSMT_RSCH_STR_DE DATETIME COMMENT '위탁 연구 시작 일자' 
        , CSMT_RSCH_END_DE DATETIME COMMENT '위탁 연구 종료 일자' 
    */
, PS_SBJT_RSCH_ORGN TEXT COMMENT '과제 연구 기관'
/*GRND_PS.PS_SBJT_RSCH_MBR : SBJT_ID, PRG_SN, PTC_PRG_SN*/
    /* Table -> json*/
    /*
        , SBJT_EXE_ANNL TINYINT DEFAULT 1 NOT NULL COMMENT '과제 수행 연차' 
        , SPRT_ORGN_ID VARCHAR(8) NOT NULL COMMENT '지원 기관 ID' 
        , SPRT_ORGN_ROLE_SE VARCHAR(6) NOT NULL COMMENT '지원 기관 역할 구분[AB2]' 
            --PO_COMM_CD JOIN
        , SPRT_ORGN_ROLE_SE_NM VARCHAR(6) NOT NULL COMMENT '지원 기관 역할 구분명' 
        , SPRT_ORGN_NM VARCHAR(300) NOT NULL COMMENT '지원 기관 명' 
        , ORGN_TP_SE VARCHAR(6) NOT NULL COMMENT '기관 유형 구분[PL7]' 
            --PO_COMM_CD JOIN
        , ORGN_TP_SE_NM VARCHAR(6) NOT NULL COMMENT '기관 유형 구분명' 
        , NAT_SE VARCHAR(6) NOT NULL COMMENT '국가 구분[PH1]' 
            --PO_COMM_CD JOIN
        , NAT_SE_NM VARCHAR(6) NOT NULL COMMENT '국가 구분명' 
        , ORGN_ROLE_DSCR VARCHAR(4000) COMMENT '기관 역할 설명' 
        , SPRT_END_DE DATETIME DEFAULT STR_TO_DATE('99991231','%Y%m%d') NOT NULL COMMENT '지원 종료 일자'
    */
, PS_SBJT_RSCH_MBR TEXT COMMENT '과제 연구 인력'
/*GRND_PS.PS_SBJT_ORGN_RSCH_EXE : SBJT_ID, PRG_SN, PTC_PRG_SN*/
    /* Table -> json*/
    /*
        , RSCH_ORGN_ID VARCHAR(8) NOT NULL COMMENT '연구 기관 ID' 
        , RSCH_EXE_SN INT DEFAULT 1 NOT NULL COMMENT '연구 수행 순번' 
        , EXE_PRG_SE VARCHAR(6) NOT NULL COMMENT '수행 진행 구분[AY3]' 
            --PO_COMM_CD JOIN
        , EXE_PRG_SE_NM VARCHAR(6) NOT NULL COMMENT '수행 진행 구분명' 
        , HAN_SBJT_NM VARCHAR(300) NOT NULL COMMENT '한글 과제 명' 
        , BSNS_NM VARCHAR(100) NOT NULL COMMENT '사업 명' 
        , GOVD_NM VARCHAR(300) NOT NULL COMMENT '부처 명' 
        , SORGN_NM VARCHAR(300) NOT NULL COMMENT '전문기관 명' 
        , RND_SUMM_CN VARCHAR(4000) NOT NULL COMMENT '연구개발 요약 내용' 
        , TTL_RSCH_STR_DE DATETIME NOT NULL COMMENT '총 연구 시작 일자' 
        , TTL_RSCH_END_DE DATETIME NOT NULL COMMENT '총 연구 종료 일자' 
        , RSCH_STR_DE DATETIME NOT NULL COMMENT '연구 시작 일자' 
        , RSCH_END_DE DATETIME NOT NULL COMMENT '연구 종료 일자' 
        , RSCT_AM BIGINT DEFAULT 0 NOT NULL COMMENT '연구비 금액' 
        , RSCH_ORGN_ROLE_SE VARCHAR(6) NOT NULL COMMENT '연구 기관 역할 구분[AB2]' 
            --PO_COMM_CD JOIN
        , RSCH_ORGN_ROLE_SE_NM VARCHAR(6) NOT NULL COMMENT '연구 기관 역할 구분명' 
    */
, PS_SBJT_ORGN_RSCH_EXE TEXT COMMENT '과제 기관 연구 수행'
/* GRND_PS.PS_SBJT_SPRT_ORGN : SBJT_ID, PRG_SN, PTC_PRG_SN */ 
    /* Table -> json*/
    /*
        , RSCH_ORGN_ID VARCHAR(8) NOT NULL COMMENT '연구 기관 ID' 
        , RSCH_EXE_SN INT DEFAULT 1 NOT NULL COMMENT '연구 수행 순번' 
        , EXE_PRG_SE VARCHAR(6) NOT NULL COMMENT '수행 진행 구분[AY3]' 
        , HAN_SBJT_NM VARCHAR(300) NOT NULL COMMENT '한글 과제 명' 
        , BSNS_NM VARCHAR(100) NOT NULL COMMENT '사업 명' 
        , GOVD_NM VARCHAR(300) NOT NULL COMMENT '부처 명' 
        , SORGN_NM VARCHAR(300) NOT NULL COMMENT '전문기관 명' 
        , RND_SUMM_CN VARCHAR(4000) NOT NULL COMMENT '연구개발 요약 내용' 
        , TTL_RSCH_STR_DE DATETIME NOT NULL COMMENT '총 연구 시작 일자' 
        , TTL_RSCH_END_DE DATETIME NOT NULL COMMENT '총 연구 종료 일자' 
        , RSCH_STR_DE DATETIME NOT NULL COMMENT '연구 시작 일자' 
        , RSCH_END_DE DATETIME NOT NULL COMMENT '연구 종료 일자' 
        , RSCT_AM BIGINT DEFAULT 0 NOT NULL COMMENT '연구비 금액' 
        , RSCH_ORGN_ROLE_SE VARCHAR(6) NOT NULL COMMENT '연구 기관 역할 구분[AB2]' 
    */
, PS_SBJT_SPRT_ORGN TEXT COMMENT '과제 지원 기관'
/*GRND_PS.PS_FRUT_SBJT_CLCT : SBJT_ID */
    /* Table -> json */
    /* 
        FRUT_ID VARCHAR(16) NOT NULL COMMENT '성과 ID' 
        , SBJT_ID VARCHAR(8) NOT NULL COMMENT '과제 ID' 
        , FRUT_TP_CD VARCHAR(5) NOT NULL COMMENT '성과 유형 코드' 
            -- PS_STDD_FRUT_TP_CD JOIN
        , FRUT_TP_CD_NM VARCHAR(500) NOT NULL COMMENT '성과 유형 코드명' 
    */
, PS_FRUT_SBJT_CLCT TEXT COMMENT '과제 수집 성과'
/*GRND_PS.PS_SBJT_TECL : SBJT_ID, PRG_SN, PTC_PRG_SN */
    /* 국가과학기술표준분류에 해당 하는 데이터만 JOIN, SBJT_TECL_GRP_CD = 'T0001'
    우선 순위가 가장 높은 1개의 데이터만 가져옴*/
, SBJT_TECL_CD VARCHAR(10) DEFAULT 'UK9999' COMMENT '과제 기술분류 코드' 
    /* PO_TECL_CD join*/
, SBJT_TECL_NM VARCHAR(10) DEFAULT '알수없음' COMMENT '과제 기술분류명' 
);
ALTER TABLE LIMENET_ANAL.DOCUMENT COMMENT '분석용 과제 정보';
