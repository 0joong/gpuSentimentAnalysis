import time
import pandas as pd
import re
from urllib.parse import quote
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup

def get_post_links(driver, search_query):
    """
    검색 결과 페이지에 접속하여 상위 5개 게시물의 링크를 수집합니다.
    """
    post_links = []
    try:
        encoded_query = quote(search_query)
        search_url = f"https://coolenjoy.net/bbs/search4.php?onetable=28&bo_table=28&sfl=wr_subject&stx={encoded_query}"
        
        driver.get(search_url)
        print(f"검색 결과 페이지로 직접 이동: {search_url}")

        list_container_selector = "ul.na-table"
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, list_container_selector))
        )
        
        soup = BeautifulSoup(driver.page_source, 'lxml')
        search_list_container = soup.select_one(list_container_selector)
        
        if not search_list_container:
            print("오류: 검색 결과 컨테이너(ul.na-table)를 찾을 수 없습니다.")
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
        print(f"총 {len(post_links)}개의 고유한 게시물 링크를 수집했습니다.")
        
    except Exception as e:
        print(f"게시물 링크 수집 중 오류 발생: {e}")

    return post_links

def get_text_from_post(driver, url):
    """
    개별 게시물 페이지에 접속하여 본문과 댓글 텍스트를 수집합니다.
    (답글에서 @닉네임 답글 부분만 제거)
    """
    texts = []
    try:
        driver.get(url)
        
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "bo_v_atc")))
        soup = BeautifulSoup(driver.page_source, 'lxml')

        # 본문 수집
        content_element = soup.find('div', id='bo_v_atc')
        if content_element:
            view_content = content_element.find('div', class_='view-content')
            if view_content:
                content_text = view_content.get_text(separator='\n', strip=True)
                texts.append({'text': content_text})
        
        # 댓글 수집
        comment_elements = soup.select('#bo_vc div.cmt_contents')
        for comment in comment_elements:
            secret_icon = comment.find('span', class_='na-icon na-secret')
            if secret_icon:
                continue

            comment_text = comment.get_text(separator='\n', strip=True)

            # *** 수정된 부분: '@닉네임 답글' 패턴을 정확히 제거 ***
            # 정규표현식을 사용하여 '@닉네임 답글'과 그 뒤의 공백까지 제거합니다.
            cleaned_text = re.sub(r'^@\S+\s+답글\s*', '', comment_text.strip())

            if cleaned_text: # 텍스트가 남아있는 경우에만 추가
                texts.append({'text': cleaned_text})
                
    except Exception as e:
        print(f"'{url}' 페이지 처리 중 오류 발생: {e}")
        
    return texts

def main():
    """
    메인 실행 함수
    """
    query = input("쿨앤조이 자유게시판에서 검색할 단어를 입력하세요: ")
    if not query:
        print("검색어가 입력되지 않았습니다.")
        return

    all_texts = []
    chrome_options = Options()
    
    # *** 수정된 부분: Headless 모드 활성화 ***
    chrome_options.add_argument("--headless")
    
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36")
    chrome_options.add_argument('--log-level=3') # 불필요한 로그 메시지 숨기기

    driver = None
    try:
        driver = webdriver.Chrome(options=chrome_options)
        
        post_links = get_post_links(driver, query)
        
        if not post_links:
            print("수집할 게시물이 없습니다. 프로그램을 종료합니다.")
            return
            
        print("\n상세 페이지 크롤링을 시작합니다.")
        
        for i, link in enumerate(post_links):
            print(f"[{i+1}/{len(post_links)}] '{link}' 처리 중...")
            texts_from_post = get_text_from_post(driver, link)
            if texts_from_post:
                all_texts.extend(texts_from_post)
                print(f"  > 텍스트 {len(texts_from_post)}개 수집 완료.")
            else:
                print("  > 수집된 텍스트가 없습니다.")
            time.sleep(1)

    except Exception as e:
        print(f"전체 작업 중 오류가 발생했습니다: {e}")
    finally:
        if driver:
            driver.quit()
            print("\nWebDriver 종료.")
            
    if all_texts:
        df = pd.DataFrame(all_texts)
        output_filename = f"coolenjoy_{query}_crawled.csv"
        df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        print(f"\n✅ 성공! 총 {len(all_texts)}개의 텍스트를 '{output_filename}' 파일로 저장했습니다.")
    else:
        print("\n❌ 수집된 데이터가 없습니다.")

if __name__ == "__main__":
    main()

