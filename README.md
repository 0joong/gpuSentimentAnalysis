# 딥러닝 기반 GPU 여론 분석 모델

### 📌 프로젝트 개요
본 프로젝트는 한국어 텍스트(리뷰, 댓글 등)의 감성(긍정, 부정, 중립)을 효과적으로 분류하는 딥러닝 모델을 개발하여 커뮤니티의 최신 GPU 여론 감성을 분석하는 것을 목표로 합니다.

### 📂 프로젝트 구조.
```
├── model/                  # 학습된 모델과 전처리기 객체를 저장하는 폴더
│   ├── model.h5            # 학습된 딥러닝 모델(구조 및 가중치) 파일
│   └── tokenizer.pickle    # 텍스트 전처리를 위한 단어 사전(Tokenizer) 객체
├── coolnjoyCrawler.py        # 커뮤니티 리뷰 데이터 수집을 위한 웹 크롤러
└── app.py                    # Streamlit 기반의 감성 분석 웹 애플리케이션 실행 파일
```
### 시작하기
1. model.h5 다운로드   
[model.h5 파일을 다운로드](https://google.com, "google link") 받은 후 프로젝트의 model 폴더에 넣어줍니다.


2. 환경 설정
라이브러리 설치먼저 프로젝트에 필요한 라이브러리들을 설치합니다.
```
pip install pandas numpy scikit-learn tensorflow konlpy tqdm matplotlib seaborn
```
3. 실행
```
streamlit run app.py
```
## 모델 학습 과정 흐름도
![image](https://github.com/user-attachments/assets/d2790d9f-1e5b-4c92-acec-b11cea0b9ef2)

## 웹 애플리케이션 동작 방식
![image](https://github.com/user-attachments/assets/e7e01be2-77ad-4491-b76d-8ff1b733f5a6)



### 데이터 수집
하드웨어 커뮤니티인 '쿨엔조이'의 그래픽카드 게시판에서 약 60만 개의 게시글 및 댓글 데이터를 수집했습니다.


### 데이터 라벨링
방대한 양의 텍스트를 사람이 직접 분류하는 것은 현실적으로 어렵습니다.    
이 문제를 해결하기 위해 LG의 Exaone과 Google의 Gemma3 거대 언어 모델(LLM)로 교차 검증을 수행했습니다.    
두 모델이 동일한 감성 레이블을 부여한 데이터만을 선별하여 사용함으로써, 높은 품질의 학습 데이터셋을 구축했습니다.   
교차검증 완료 후 데이터는 약 20만개 입니다.


### 텍스트 전처리
사전 학습된 `klue/bert-base` 모델의 성능을 극대화하기 위해, 모델이 학습될 때와 동일한 방식의 전처리를 적용합니다.

1. **정규식을 이용한 노이즈 제거**: 기본적인 정제 작업을 수행합니다.
2. **중복치 제거**: 데이터셋 내의 중복된 텍스트 샘플을 제거합니다.
3. **BERT 토크나이저 적용**: Hugging Face의 `AutoTokenizer`를 사용하여 텍스트를 토큰화합니다. 이 과정에는 다음과 같은 작업이 포함됩니다.
    - **WordPiece Tokenization**: 단어를 더 작은 의미 단위(Subword)로 분할합니다. 
    - **Special Tokens 추가**: 문장의 시작(`[CLS]`)과 끝(`[SEP]`)을 알리는 특수 토큰을 추가합니다.
    - **정수 인코딩 및 패딩**: 토큰들을 고유 정수로 변환하고, 모든 입력의 길이를 통일(Padding)하며, 어텐션 마스크(Attention Mask)를 생성합니다.

### 모델 아키텍처
본 프로젝트는 사전 학습된 언어 모델인 klue/bert-base를 감성 분석 작업에 맞게 미세 조정하여 사용했습니다.    
전체적인 데이터 처리 흐름은 다음과 같습니다. 
1. **klue/bert-base 모델**: 전처리된 텍스트를 입력받아 풍부한 문맥 정보가 담긴 고차원 벡터로 변환합니다.
2. **분류기 (Classifier)**: klue/bert-base가 추출한 핵심 의미 벡터([CLS] 토큰)를 입력받습니다.
3. **과적합 방지 및 최종 출력**: Dropout을 거친 후, Softmax 함수를 통해 최종적으로 각 감성 클래스 확률을 출력합니다.

## 모델 훈련
### 정확도 및 손실 그래프
![image](https://github.com/user-attachments/assets/5cff0639-0224-4d4c-a85e-410fddfcdda2)
모델은 1~2 Epoch에서 최적의 성능에 도달했으며, 그 이후부터는 훈련 데이터에만 과도하게 최적화되는 과적합이 발생했습니다.   
코드에 EarlyStopping과 ModelCheckpoint를 설정해두었기 때문에, 검증 성능이 가장 좋았던 시점의 모델이 최종 모델로 저장되었습니다.   
### 분류 리포트
![image](https://github.com/user-attachments/assets/96f9a4b1-fcbd-4f7c-a30e-71959f4d2c28)   
전반적으로 모델이 약 88% 수준의 높은 정확도(Accuracy)를 보이며, 모든 클래스를 86% 이상의 F1-Score로 안정적으로 분류하고 있습니다.    
이는 모델이 특정 감성에 치우치지 않고 긍정, 부정, 중립을 모두 준수하게 학습했음을 의미합니다.
## 사용법
### 초기화면
![image](https://github.com/user-attachments/assets/4027ff64-a82c-410d-8853-cc493c7f9103)
검색창에 원하는 그래픽카드 모델명을 입력합니다.   

### 진행중
![image](https://github.com/user-attachments/assets/58ff5578-9da5-4b9f-8bc8-7c6202fe88f6)
입력 후 버튼을 클릭하면 진행 상황을 보여주며 웹 크롤링 및 감성 분석이 시작됩니다.   

### 결과
![image](https://github.com/user-attachments/assets/6b34b59a-c983-4210-a704-028b1fd95d20)
해당 그래픽카드에 대한 감성 비율 및 세부 분석 결과가 표시됩니다.



