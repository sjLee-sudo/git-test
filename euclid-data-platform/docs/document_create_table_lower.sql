CREATE TABLE limenet_analysis.document
(
 doc_id VARCHAR(30) PRIMARY KEY COMMENT '문서 ID'
/*GRND_PS.PS_SBJT_PRG_PTC */
, sbjt_id VARCHAR(8) NOT NULL COMMENT '과제 ID' 
, prg_sn INT  DEFAULT 1 NOT NULL COMMENT '진행 순번' 
, ptc_prg_sn INT DEFAULT 1 NOT NULL COMMENT '세부 진행 순번' 
/*GRND_PS.PS_SBJT : SBJT_ID*/
, rnd_sbjt_no VARCHAR(16) NULL COMMENT '연구개발 과제 번호' 
, han_sbjt_nm VARCHAR(300) NULL COMMENT '한글 과제 명' 
, eng_sbjt_nm VARCHAR(300) NULL COMMENT '영문 과제 명' 
, ovrl_sbjt_id VARCHAR(8) COMMENT '총괄 과제 ID' 
, han_ovrl_rnd_nm VARCHAR(300) COMMENT '한글 총괄 연구개발 명' 
, eng_ovrl_rnd_nm VARCHAR(300) COMMENT '영문 총괄 연구개발 명' 
, bsns_yy VARCHAR(4) NULL COMMENT '사업 년도' 
, sorgn_bsns_cd VARCHAR(7) NULL COMMENT '전문기관 사업 코드' 
    /*총괄/세부/일반*/
, sbjt_propl_strc_se VARCHAR(6) NULL COMMENT '과제 추진 체계 구분[AD4]' 
    /*PO_COMM_CD JOIN*/
, sbjt_propl_strc_se_nm VARCHAR(6) NULL COMMENT '과제 추진 체계 구분명'
    /* 단계/일반*/
, rsch_prid_cpst_se VARCHAR(6) NULL COMMENT '연구 기간 구성 구분[AB5]' 
    /*PO_COMM_CD JOIN*/
, rsch_prid_cpst_se_nm VARCHAR(6) NULL COMMENT '연구 기간 구성 구분명' 
    /*개념계획서,과제조정,신청용 연구개발계획서,차단계 연구개발계획서,협약변경,협약용 연구개발계획서*/
, rcve_pldoc_tp_se VARCHAR(6) NULL COMMENT '접수 계획서 유형 구분[AS5]' 
    /*PO_COMM_CD JOIN*/
, rcve_pldoc_tp_se_nm VARCHAR(6) NULL COMMENT '접수 계획서 유형 구분명' 
    /*원천기술형/혁신제품형/해당사항없음*/
, sbjt_dvlm_tp_se VARCHAR(6) NULL COMMENT '과제 개발 유형 구분[AA7]' 
    /*PO_COMM_CD JOIN*/
, sbjt_dvlm_tp_se_nm VARCHAR(6) NULL COMMENT '과제 개발 유형 구분명' 
, ovrs_nopen_yn VARCHAR(1) DEFAULT 'Y' NULL COMMENT '대외 비공개 여부' 
, prg_sbjt_exe_annl TINYINT DEFAULT 1 NULL COMMENT '진행 과제 수행 연차' 
, prg_bsns_yy VARCHAR(4) NULL COMMENT '진행 사업 년도' 
, prg_sorgn_bsns_cd VARCHAR(7) NULL COMMENT '진행 전문기관 사업 코드' 
    /*개발종(완)료,개발중,계속,과제접수결과,선정,선정평가결과,신청/접수,연구개발계획서제출 대상,중단,탈락,협약체결결과,협약포기*/
, sbjt_prg_se VARCHAR(6) NULL COMMENT '과제 진행 구분[AD3]' 
    /*PO_COMM_CD JOIN*/
, sbjt_prg_se_nm VARCHAR(6) NULL COMMENT '과제 진행 구분명' 
    /*정상과제,모의과제*/
, sbjt_crt_se VARCHAR(6) NULL COMMENT '과제 생성 구분[BP5]' 
    /*PO_COMM_CD JOIN*/
, sbjt_crt_se_nm VARCHAR(6) NULL COMMENT '과제 생성 구분명' 
    /*계획서 작성중,기관담당자 제출반려,기관담당자 제출승인,접수마감,접수반려,접수취소,제출완료,제출유예*/
, last_rcve_stt_se VARCHAR(6) NULL COMMENT '최종 접수 상태 구분[AO4]' 
    /*PO_COMM_CD JOIN*/
, last_rcve_stt_se_nm VARCHAR(6) NULL COMMENT '최종 접수 상태 구분명' 
/*GRND_PS.PS_SBJT_TOT_ANNL : SBJT_ID, PRG_SN, PTC_PRG_SN*/
, rsch_str_de DATETIME NULL COMMENT '연구 시작 일자' 
, rsch_end_de DATETIME NULL COMMENT '연구 종료 일자' 
, rsch_gole_cn TEXT NULL COMMENT '연구 목표 내용' 
, rsch_rang_cn TEXT NULL COMMENT '연구 범위 내용' 
, expe_efct_cn TEXT NULL COMMENT '기대 효과 내용' 
, prspt_frut_cn TEXT NULL COMMENT '예상 성과 내용' 
/*GRND_PS.PS_SBJT_KWD : SBJT_ID, PRG_SN, PTC_PRG_SN*/
    /* 한/영 구분없이 키워드 순번대로 , 로 묶음*/
, kwd VARCHAR(1000) DEFAULT '' NULL COMMENT '키워드 명' 
/*GRND_PS.PS_SORGN_YY_BSNS : SORGN_BSNS_CD*/
, hirk_sorgn_bsns_cd VARCHAR(7) COMMENT '상위 전문기관 사업 코드' 
    /* PS_SORGN_YY_BSNS : SORGN_BSNS_CD  */
, hirk_sorgn_bsns_nm VARCHAR(100) COMMENT '상위 전문기관 사업명' 
, sorgn_id VARCHAR(5) NULL COMMENT '전문기관 ID' 
, sorgn_bsns_nm VARCHAR(100) NULL COMMENT '전문기관 사업 명' 
    /*내역사업,단위사업,세부사업,프로그램*/
, bsns_csrt_se VARCHAR(6) NULL COMMENT '사업 구조 구분[AH1]'
    /*PO_COMM_CD JOIN */
, bsns_csrt_se_nm VARCHAR(6) NULL COMMENT '사업 구조 구분명' 
    /*국제기술협력,기반조성,기술개발,사업화,인력양성,지역산업,표준화,학술*/
, bsns_tp_se VARCHAR(6) COMMENT '사업 유형 구분[AH6]' 
    /*PO_COMM_CD JOIN*/
, bsns_tp_se_nm VARCHAR(6) COMMENT '사업 유형 구분명' 
, sbjt_sprt_bud_am BIGINT DEFAULT 0 NULL COMMENT '과제 지원 예산 금액' 
, plnn_eval_bud_am BIGINT DEFAULT 0 NULL COMMENT '기획 평가 예산 금액' 
, dlgt_ectn_bsns_yn VARCHAR(1) DEFAULT 'N' NULL COMMENT '위임 예외 사업 여부' 
    /*계속,분할계속,신규,신규,일반계속,종료,종료,통합계속*/
, bsns_prg_se VARCHAR(6) COMMENT '사업 진행 구분[AH7]' 
    /*PO_COMM_CD JOIN*/
, bsns_prg_se_nm VARCHAR(6) COMMENT '사업 진행 구분명' 
    /*GRND_PS.PS_SORGN  : SORGN_ID*/
, hirk_sorgn_id VARCHAR(5) COMMENT '상위 전문기관 ID' 
    /*PS_SORGN JOIN*/
, hirk_sorgn_nm VARCHAR(5) COMMENT '상위 전문기관명' 
, sorgn_nm VARCHAR(300) NULL COMMENT '전문기관 명' 
    /*범(다)부처사업단,전문기관,전문기관 소속 사업단*/
, sorgn_role_se VARCHAR(6) NULL COMMENT '전문기관 역할 구분[AS3]' 
    /*PO_COMM_CD JOIN*/
, sorgn_role_se_nm VARCHAR(6) NULL COMMENT '전문기관 역할 구분명' 
    /*경찰청,고용노동부,공정거래위원회,과학기술부,과학기술정보통신부,교육과학기술부,교육부,국무조정실및국무총리비서실,국무총리실,국방부,국토교통부,기상청,기획재정부,노동부,농림축산식품부,농촌진흥청,문화관광부,문화재청,문화체육관광부,미래창조과학부,방송통신위원회,방위사업청,범부처,법무부,법제처,보건복지가족부,보건복지부,산림청,산업통상자원부,새만금개발청,소방청,식품의약품안전처,여성가족부,여성부,외교부,원자력안전위원회,인사혁신처,정보통신부,중소기업청,중소벤처기업부,지식경제부,통일부,특허청,해양경찰청,해양수산부,행정안전부,행정중심복합도시건설청,환경부*/
, blng_govd_se VARCHAR(6) NULL COMMENT '소속 부처 구분[AR4]' 
    /*PO_COMM_CD JOIN*/
, blng_govd_se_nm VARCHAR(6) NULL COMMENT '소속 부처 구분명' 
, orgn_id VARCHAR(8) NULL COMMENT '기관 ID' 
/*GRND_PS.PS_SBJT_RSCH_ORGN : SBJT_ID, PRG_SN, PTC_PRG_SN*/
    /* Table -> json*/
    /*
        , SBJT_EXE_ANNL TINYINT DEFAULT 1 NULL COMMENT '과제 수행 연차' 
        , RSCH_ORGN_ID VARCHAR(8) NULL COMMENT '연구 기관 ID' 
        , RSCH_ORGN_ROLE_SE VARCHAR(6) NULL COMMENT '연구 기관 역할 구분[AB2]' 
            -- PO_COMM_CD JOIN
        , RSCH_ORGN_ROLE_SE_NM VARCHAR(6) NULL COMMENT '연구 기관 역할 구분명' 
        , AGRT_ORGN_ID VARCHAR(8) NULL COMMENT '협약 기관 ID' 
        , RSCH_ORGN_NM VARCHAR(300) NULL COMMENT '연구 기관 명' 
        , ORGN_STP_PURS_SE VARCHAR(6) NULL COMMENT '기관 설립 목적 구분[PL9]' 
            -- PO_COMM_CD JOIN
        , ORGN_STP_PURS_SE_NM VARCHAR(6) NULL COMMENT '기관 설립 목적 구분명' 
        , ORGN_TP_SE VARCHAR(6) NULL COMMENT '기관 유형 구분[PL7]' 
            -- PO_COMM_CD JOIN
        , ORGN_TP_SE VARCHAR(6) NULL COMMENT '기관 유형 구분명' 
        , NAT_SE VARCHAR(6) NULL COMMENT '국가 구분[PH1]' 
            -- PO_COMM_CD JOIN
        , NAT_SE_NM VARCHAR(6) NULL COMMENT '국가 구분명' 
        , LOC_ZNE_SE VARCHAR(6) NULL COMMENT '소재지 지역 구분[PH8]' 
            -- PO_COMM_CD JOIN
        , LOC_ZNE_SE_NM VARCHAR(6) NULL COMMENT '소재지 지역 구분명' 
        , CSMT_RSCH_STR_DE DATETIME COMMENT '위탁 연구 시작 일자' 
        , CSMT_RSCH_END_DE DATETIME COMMENT '위탁 연구 종료 일자' 
    */
, ps_sbjt_rsch_orgn TEXT COMMENT '과제 연구 기관'
/*GRND_PS.PS_SBJT_RSCH_MBR : SBJT_ID, PRG_SN, PTC_PRG_SN*/
    /* Table -> json*/
    /*
        , SBJT_EXE_ANNL TINYINT DEFAULT 1 NULL COMMENT '과제 수행 연차' 
        , SPRT_ORGN_ID VARCHAR(8) NULL COMMENT '지원 기관 ID' 
        , SPRT_ORGN_ROLE_SE VARCHAR(6) NULL COMMENT '지원 기관 역할 구분[AB2]' 
            --PO_COMM_CD JOIN
        , SPRT_ORGN_ROLE_SE_NM VARCHAR(6) NULL COMMENT '지원 기관 역할 구분명' 
        , SPRT_ORGN_NM VARCHAR(300) NULL COMMENT '지원 기관 명' 
        , ORGN_TP_SE VARCHAR(6) NULL COMMENT '기관 유형 구분[PL7]' 
            --PO_COMM_CD JOIN
        , ORGN_TP_SE_NM VARCHAR(6) NULL COMMENT '기관 유형 구분명' 
        , NAT_SE VARCHAR(6) NULL COMMENT '국가 구분[PH1]' 
            --PO_COMM_CD JOIN
        , NAT_SE_NM VARCHAR(6) NULL COMMENT '국가 구분명' 
        , ORGN_ROLE_DSCR VARCHAR(4000) COMMENT '기관 역할 설명' 
        , SPRT_END_DE DATETIME DEFAULT STR_TO_DATE('99991231','%Y%m%d') NULL COMMENT '지원 종료 일자'
    */
, ps_sbjt_rsch_mbr TEXT COMMENT '과제 연구 인력'
/*GRND_PS.PS_SBJT_ORGN_RSCH_EXE : SBJT_ID, PRG_SN, PTC_PRG_SN*/
    /* Table -> json*/
    /*
        , RSCH_ORGN_ID VARCHAR(8) NULL COMMENT '연구 기관 ID' 
        , RSCH_EXE_SN INT DEFAULT 1 NULL COMMENT '연구 수행 순번' 
        , EXE_PRG_SE VARCHAR(6) NULL COMMENT '수행 진행 구분[AY3]' 
            --PO_COMM_CD JOIN
        , EXE_PRG_SE_NM VARCHAR(6) NULL COMMENT '수행 진행 구분명' 
        , HAN_SBJT_NM VARCHAR(300) NULL COMMENT '한글 과제 명' 
        , BSNS_NM VARCHAR(100) NULL COMMENT '사업 명' 
        , GOVD_NM VARCHAR(300) NULL COMMENT '부처 명' 
        , SORGN_NM VARCHAR(300) NULL COMMENT '전문기관 명' 
        , RND_SUMM_CN VARCHAR(4000) NULL COMMENT '연구개발 요약 내용' 
        , TTL_RSCH_STR_DE DATETIME NULL COMMENT '총 연구 시작 일자' 
        , TTL_RSCH_END_DE DATETIME NULL COMMENT '총 연구 종료 일자' 
        , RSCH_STR_DE DATETIME NULL COMMENT '연구 시작 일자' 
        , RSCH_END_DE DATETIME NULL COMMENT '연구 종료 일자' 
        , RSCT_AM BIGINT DEFAULT 0 NULL COMMENT '연구비 금액' 
        , RSCH_ORGN_ROLE_SE VARCHAR(6) NULL COMMENT '연구 기관 역할 구분[AB2]' 
            --PO_COMM_CD JOIN
        , RSCH_ORGN_ROLE_SE_NM VARCHAR(6) NULL COMMENT '연구 기관 역할 구분명' 
    */
, ps_sbjt_orgn_rsch_exe TEXT COMMENT '과제 기관 연구 수행'
/* GRND_PS.PS_SBJT_SPRT_ORGN : SBJT_ID, PRG_SN, PTC_PRG_SN */ 
    /* Table -> json*/
    /*
        , SBJT_EXE_ANNL NUMBER(2, 0) DEFAULT 1 NOT NULL 
        , SPRT_ORGN_ID VARCHAR2(8 BYTE) NOT NULL 
        , SPRT_ORGN_ROLE_SE VARCHAR2(6 BYTE) NOT NULL 
        , SPRT_ORGN_NM VARCHAR2(300 BYTE) NOT NULL 
        , ORGN_TP_SE VARCHAR2(6 BYTE) NOT NULL 
        , NAT_SE VARCHAR2(6 BYTE) NOT NULL 
        , ORGN_ROLE_DSCR VARCHAR2(4000 BYTE) 
        , SPRT_END_DE DATE DEFAULT TO_DATE('99991231','YYYYMMDD') NOT NULL 
        , LAST_YN VARCHAR2(1 BYTE) DEFAULT 'Y' NOT NULL 
        , FRST_REG_MBR_ID VARCHAR2(8 BYTE) NOT NULL 
        , FRST_REG_DT DATE DEFAULT SYSDATE NOT NULL 
        , LAST_MODF_MBR_ID VARCHAR2(8 BYTE) NOT NULL 
        , LAST_MODF_DT DATE DEFAULT SYSDATE NOT NULL 
    */
, ps_sbjt_sprt_orgn TEXT COMMENT '과제 지원 기관'
/*GRND_PS.PS_FRUT_SBJT_CLCT : SBJT_ID */
    /* Table -> json */
    /* 
        FRUT_ID VARCHAR(16) NULL COMMENT '성과 ID' 
        , SBJT_ID VARCHAR(8) NULL COMMENT '과제 ID' 
        , FRUT_TP_CD VARCHAR(5) NULL COMMENT '성과 유형 코드' 
            -- PS_STDD_FRUT_TP_CD JOIN
        , FRUT_TP_CD_NM VARCHAR(500) NULL COMMENT '성과 유형 코드명' 
    */
, ps_frut_sbjt_clct TEXT COMMENT '과제 수집 성과'
/*GRND_PS.PS_SBJT_TECL : SBJT_ID, PRG_SN, PTC_PRG_SN */
    /* 국가과학기술표준분류에 해당 하는 데이터만 JOIN, SBJT_TECL_GRP_CD = 'T0001'
    우선 순위가 가장 높은 1개의 데이터만 가져옴*/
, sbjt_tecl_cd VARCHAR(10) NOT NULL DEFAULT 'UK9999' COMMENT '과제 기술분류 코드' 
    /* PO_TECL_CD join*/
, sbjt_tecl_nm VARCHAR(10) NOT NULL DEFAULT '알수없음' COMMENT '과제 기술분류명' 
, reg_dt DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '분석 자료 생성일자' 
);
ALTER TABLE limenet_analysis.document COMMENT '분석용 과제 정보';
