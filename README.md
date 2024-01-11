### <div align="center">📺 LG헬로비전 VOD 개인화 추천 프로젝트

헬로 TV 사용자를 위한 VOD 추천 서비스
- LG 헬로비전 서비스 특성을 반영한 VOD 추천 서비스 웹기반 개발
- 기존 서비스 및 데이터 분석을 통해 우리만의 서비스 방향성 설정하여 새로운 VOD 서비스 제공
- TV에서의 사용성을 고려한 UI 구현

### 목차

### 개요
![Group 79](https://github.com/yOukyonG/LGhv_VOD_REC/assets/122434675/65d5abd0-8e9c-464d-9446-70c53edd208b)

- **팀 이름** : Hello00 팀
- **팀원 소개** : 공유경, 김명현, 김은혜, 박채나, 황성주 (5명)
- **프로젝트 기간** : 2023.10 ~ 2023.12.29
- **웹사이트 주소** : httsp://hello00.net
- **서비스 배포** : 2023.11 ~ 2023.12
- **개발 도구** :
  - (Data) Python, RDS
  - (BE) Django, AWS, Airflow
  - (FE) JavaScript, React, Figma
  - (Communication) Github, Notion, Slack
    
	<div align="center">
	<img src="https://img.shields.io/badge/Java-007396?style=flat&logo=Java&logoColor=white" />
	<img src="https://img.shields.io/badge/HTML5-E34F26?style=flat&logo=HTML5&logoColor=white" />
	<img src="https://img.shields.io/badge/CSS3-1572B6?style=flat&logo=CSS3&logoColor=white" />
</div>
  
### 프로젝트 소개
#### 1. 기존 서비스 분석
**1) 다양한 연령층을 위한 콘텐츠 보유**
- 넷플릭스, 유튜브, 디즈니+, 아이들나라, 지역채널 등 다양한 연령층, 가족 구성원
- 중장년층의 높은 비중
  
**2) 지역 특화 콘텐츠**
- 가입 가능한 지역 특정되어 있으며 제철장터, 지역 방송 등 지역 특화 콘텐츠 존재

 
#### 2. 사용자 분석
데이터 EDA 를 통한 서비스 사용자 분석 진행

**1) 시간대별 시청 형태 (키즈, 성인)**
   키즈와 성인 시청이 겹치는 시간대 존재
   -> 이용자 특성에 따른 시간대별 추천 결과 필터링 필요성 확인

**2) 장르별 시청 분석**
   주 고객층인 중장년층 선호 장르에 대한 낮은 시청 비중
   -> 중장년층을 위한 서비스 확대 필요

**3) 사용자 시청 분석**
   사용자 대부분이 VOD 시청 전환율이 낮은 군집에 포함
   -> 시청 전환율 개선을 위한 추천 서비스 고도화 필요

 
### 3. 서비스 핵심 기능 소개
**1) Simple 모드**
![슬라이드11](https://github.com/yOukyonG/LGhv_VOD_REC/assets/122434675/c2744afb-02f5-43b2-9aa0-5a40bd4acda4)

**2) 키즈 / 성인 필터링 기능**
![슬라이드12](https://github.com/yOukyonG/LGhv_VOD_REC/assets/122434675/e51c581f-4afd-4649-a7db-1bad87a1c6b7)

![슬라이드13](https://github.com/yOukyonG/LGhv_VOD_REC/assets/122434675/bdef52fc-f21b-4935-a369-e403602c47eb)

**3) 재추천 기능**
![슬라이드15](https://github.com/yOukyonG/LGhv_VOD_REC/assets/122434675/4b206523-5344-483c-87a6-ff82aab9777a)

### 4. 서비스 개발 과정
1) 데이터 전처리
   - 세부 정보 크롤링 : 세부장르, 시청 연령, 포스터 이미지 URL, 개봉일/ 업로드 날짜
   - 프로그램 이름 추출
   - 시간 관련 변수 생성
   - 이벤트성 데이터 제거
   - Subsr_id, Kids 구분 컬럼 추가
     
2) 데이터 베이스 ERD 설계
   ![KakaoTalk_20240111_160213447](https://github.com/yOukyonG/LGhv_VOD_REC/assets/122434675/3f2a4bc6-cf14-4bf8-92e2-e92208d4d816)


3) 데이터 모델링
- 유저 시청 패턴에 따른 군집화를 진행한 후, 각 유저별 추천 모델 적용
  ![슬라이드22](https://github.com/yOukyonG/LGhv_VOD_REC/assets/122434675/93887800-9a86-4e86-af69-30a0534339fe)
 ![슬라이드24](https://github.com/yOukyonG/LGhv_VOD_REC/assets/122434675/aacb5bce-0cc3-4b12-b351-a1bdd9db8f9f)

4) BE

5) CI/CD
6) 
   
