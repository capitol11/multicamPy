import os
from urllib.request import urlretrieve
from selenium import webdriver
from bs4 import BeautifulSoup as bs
import time
import pandas as pd

from 프로젝트.술이미지크롤링.이미지크롤링 import Base_URL, get_base_url

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

data = pd.read_csv('C:/Users/Soohyun/Desktop/AI 프로젝트 1조/한국농수산식품유통공사_전통주 정보_20210914.csv', encoding='cp949')
alcohol_names = data['전통주명'].tolist()
alcohol_names = alcohol_names[1:]

total_urls = []

driver = webdriver.Chrome(r'C:/pydata/chromedriver')


def createFolder(directory_name):
    try:
        if not os.path.exists(directory_name):
            path = './img/' + directory_name
            os.mkdir(path)
    except OSError:
        pass
        #print('Error: ' + directory_name)
    except FileNotFoundError:
        print('Error: 파일 경로를 알 수 없습니다.')


def search전통주(url, part_url):
    driver.get(url + part_url)
    time.sleep(1) # 페이지 열리는거 잠시 기다림

    html_source = driver.page_source
    soup = bs(html_source, "html.parser")  # 이미지 스크롤 해서 이미지 더 가져오기

    img_urls = []
    image = soup.find_all('img', limit= 50)

    cnt = 0
    for i in image:
        cnt+=1
        if cnt > 3 and 'data:image' not in i.get('src'):  # 이상한 배너/링크 사전에 제거
            img_urls.append(i.get('src'))

    print(img_urls)

    for index, link in enumerate(img_urls) :
        urlretrieve(link, f'./img/{part_url}/{part_url}_{index}.jpg')
    print('다운로드 완료.')


base_url = Base_URL()

url = str(get_base_url()).split('%')[0] # get URL from 이미지크롤링.py
print(url)

total_urls = ["까메오 막걸리", "기다림 34"]
for i in alcohol_names:
    createFolder(i)

for i in total_urls:
    search전통주(url, i)
