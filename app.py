import streamlit as st
import pandas as pd
import numpy as np
import re
import time
import os
from urllib.parse import quote
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup
import joblib
import matplotlib.pyplot as plt
from konlpy.tag import Okt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

# ==============================================================================
# 쿨앤조이 크롤러 함수
# ==============================================================================
def get_post_links(driver, search_query):
    """
    검색 결과 페이지에 접속하여 상위 5개 게시물의 링크를 수집합니다.
    """
    post_links = []
    try:
        encoded_query = quote(search_query)
        search_url = f"https://coolenjoy.net/bbs/search4.php?onetable=28&bo_table=28&sfl=wr_subject&stx={encoded_query}"
        
        driver.get(search_url)
        st.write(f"ℹ️ 검색 결과 페이지로 이동: {search_url}")

        list_container_selector = "ul.na-table"
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, list_container_selector))
        )
        
        soup = BeautifulSoup(driver.page_source, 'lxml')
        search_list_container = soup.select_one(list_container_selector)
        
        if not search_list_container:
            st.warning("⚠️ 검색 결과 컨테이너를 찾을 수 없습니다.")
            return []

        link_tags = search_list_container.select('div.na-item a')
        
        for tag in link_tags:
            href = tag.get('href')
            if href and 'wr_id' in href:
                match = re.search(r'wr_id=(\d+)', href)
                if match:
                    wr_id = match.group(1)
                    post_url = f"https://coolenjoy.net/bbs/28/{wr_id}"
                    if post_url not in post_links:
                        post_links.append(post_url)
        
        post_links = post_links[:5]
        st.write(f"ℹ️ 총 {len(post_links)}개의 게시물 링크를 수집했습니다.")
        
    except Exception as e:
        st.error(f"❌ 게시물 링크 수집 중 오류: {e}")

    return post_links

def get_text_from_post(driver, url):
    """
    개별 게시물 페이지에 접속하여 본문과 댓글 텍스트를 수집합니다.
    """
    texts = []
    try:
        driver.get(url)
        
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "bo_v_atc")))
        soup = BeautifulSoup(driver.page_source, 'lxml')

        content_element = soup.find('div', id='bo_v_atc')
        if content_element:
            view_content = content_element.find('div', class_='view-content')
            if view_content:
                content_text = view_content.get_text(separator='\n', strip=True)
                texts.append({'text': content_text})
        
        comment_elements = soup.select('#bo_vc div.cmt_contents')
        for comment in comment_elements:
            secret_icon = comment.find('span', class_='na-icon na-secret')
            if secret_icon:
                continue

            comment_text = comment.get_text(separator='\n', strip=True)
            cleaned_text = re.sub(r'^@\S+\s+답글\s*', '', comment_text.strip())

            if cleaned_text:
                texts.append({'text': cleaned_text})
                
    except Exception as e:
        st.warning(f"페이지 처리 중 오류: {url} ({e})")
        
    return texts

# ==============================================================================
# 데이터 전처리 및 감성 분석 함수
# ==============================================================================
def preprocess_texts(df):
    """
    모델 학습 시와 동일한 방법으로 텍스트 데이터를 전처리합니다.
    """
    # 1. 한글 및 공백 제외 모든 문자 제거
    df['text'] = df['text'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", regex=True)
    # 2. 빈 값(empty string)을 NaN으로 변경 후 제거
    df['text'].replace('', np.nan, inplace=True)
    df.dropna(subset=['text'], inplace=True)
    
    # 3. 토큰화 및 불용어 제거
    okt = Okt()
    stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다','되다','있다','이다','그','저','것']
    
    tokenized_data = []
    for sentence in tqdm(df['text']):
        tokenized = okt.morphs(str(sentence), stem=True)
        stopwords_removed = [word for word in tokenized if not word in stopwords]
        tokenized_data.append(stopwords_removed)
        
    df['tokens'] = tokenized_data
    return df

def predict_sentiment(texts, model, tokenizer, max_len=100):
    """
    전처리된 텍스트를 입력받아 감성을 예측합니다.
    """
    # 예측을 위해 토큰화된 리스트를 다시 문자열로 합침
    sequences = tokenizer.texts_to_sequences(texts.apply(lambda x: ' '.join(x)))
    X = pad_sequences(sequences, maxlen=max_len, padding='pre')
    preds = model.predict(X, verbose=0)
    labels = ['긍정', '부정', '중립']
    results = [labels[np.argmax(p)] for p in preds]
    probs = [float(np.max(p)) * 100 for p in preds]
    return results, probs

# ==============================================================================
# Streamlit UI 설정
# ==============================================================================
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="GPU 커뮤니티 여론 분석", layout="wide")
st.title("그래픽카드 커뮤니티 여론 분석 대시보드")

st.sidebar.title("📊 프로젝트 소개")
st.sidebar.markdown("쿨앤조이 그래픽카드 게시판의 최신 여론을 수집하고 감성 분석을 수행합니다.")
st.sidebar.info("분석하고 싶은 그래픽카드 모델명을 입력하고 분석 시작 버튼을 누르세요.")

model_path = "./model/model.h5"
tokenizer_path = "./model/tokenizer.pickle"

@st.cache_resource
def load_models():
    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
        return None, None
    model = load_model(model_path)
    tokenizer = joblib.load(tokenizer_path)
    return model, tokenizer

model, tokenizer = load_models()

if not model or not tokenizer:
    st.error("❗ 모델 또는 토크나이저 파일을 찾을 수 없습니다. './model/' 폴더에 파일이 있는지 확인해주세요.")
else:
    query = st.text_input("🔍 분석할 그래픽카드 모델명을 입력하세요", placeholder="예: RTX 5080")

    if st.button("🚀 커뮤니티 여론 수집 및 감성 분석 시작"):
        if not query.strip():
            st.warning("❗ 검색어를 입력해주세요.")
        else:
            all_texts = []
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36")
            chrome_options.add_argument('--log-level=3')
            
            driver = None
            try:
                with st.spinner("1️⃣ 커뮤니티에서 최신 게시물을 수집하고 있습니다..."):
                    driver = webdriver.Chrome(options=chrome_options)
                    post_links = get_post_links(driver, query)
                    
                    if not post_links:
                        st.warning("수집할 게시물이 없습니다.")
                        st.stop()

                    for i, link in enumerate(post_links):
                        st.write(f"   - [{i+1}/{len(post_links)}] '{link}' 처리 중...")
                        texts_from_post = get_text_from_post(driver, link)
                        if texts_from_post:
                            all_texts.extend(texts_from_post)
                        time.sleep(1)

                st.success(f"✅ 게시물 수집 완료! 총 {len(all_texts)}개의 텍스트(본문+댓글)를 수집했습니다.")

                if not all_texts:
                    st.warning("수집된 텍스트가 없습니다. 분석을 종료합니다.")
                    st.stop()
                
                df_crawled = pd.DataFrame(all_texts)
                
                with st.spinner("2️⃣ 텍스트를 분석 가능한 형태로 전처리하고 있습니다..."):
                    df_processed = preprocess_texts(df_crawled)
                st.success("✅ 텍스트 전처리 완료!")

                with st.spinner("3️⃣ AI 모델로 감성을 예측하고 있습니다..."):
                    preds, probs = predict_sentiment(df_processed['tokens'], model, tokenizer)
                    df_processed['감성'] = preds
                    df_processed['신뢰도(%)'] = probs
                st.success("✅ 감성 분석 완료!")

                st.subheader(f"'{query}'에 대한 커뮤니티 여론 분석 결과")
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.markdown("#### 📊 감성 비율")
                    fig, ax = plt.subplots(figsize=(6, 5))
                    value_counts = df_processed['감성'].value_counts().reindex(['긍정', '중립', '부정']).fillna(0)
                    colors = ['#2ECC71', '#BDC3C7', '#E74C3C']
                    ax.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=90, colors=colors, textprops={'fontsize': 12})
                    ax.axis('equal')
                    st.pyplot(fig)

                with col2:
                    st.markdown("#### 🔢 감성별 텍스트 개수")
                    st.markdown(f"##### 🟢 긍정: {int(value_counts.get('긍정', 0))}개")
                    st.markdown(f"##### ⚪ 중립: {int(value_counts.get('중립', 0))}개")
                    st.markdown(f"##### 🔴 부정: {int(value_counts.get('부정', 0))}개")

                st.markdown("---")
                st.subheader("📝 세부 분석 결과")
                st.dataframe(df_processed[['text', '감성', '신뢰도(%)']].rename(columns={'text': '원문'}))

            except Exception as e:
                st.error(f"❌ 전체 작업 중 오류 발생: {e}")
            finally:
                if driver:
                    driver.quit()
