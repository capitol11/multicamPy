from selenium import webdriver
import time
import pandas as pd
from selenium.common.exceptions import StaleElementReferenceException, NoSuchElementException
from selenium.webdriver.common.keys import Keys
from urllib3.exceptions import MaxRetryError

driver = webdriver.Chrome('C:/pydata/chromedriver')
url = 'https://terms.naver.com/list.naver?cid=42726&categoryId=58635/'
driver.get(url)
time.sleep(0.5)

# get whole alcohol list
data = pd.read_csv('C:/Users/Soohyun/Desktop/AI 프로젝트 1조/한국농수산식품유통공사_전통주 정보_20210914.csv', encoding='cp949')
#alcohol_names = data['전통주명'].tolist()
#alcohol_names = ['양지백주', '황감찰']
alcohol_names = data['전통주명'][:20].tolist()
cite_names = []
#print(alcohol_names)
cnt = 0
for index, al in enumerate(alcohol_names):
    searchbox = driver.find_element_by_name('query')  # xpath, css selector X
    searchbox.send_keys(al)
    searchbox.send_keys(Keys.RETURN)
    time.sleep(3)
    try:
        page_img_text = str(driver.find_element_by_css_selector('#content > div.search_result_area > ul > li > div.info_area > div.subject > strong > a > strong').text).replace(' ', '')
        el = driver.find_element_by_id('content')
        print("content: ", el)
        # content > div:nth-child(3) > ul > li > div.info_area > div.subject > strong > a > strong

        # tag group
        # cite_name = str(driver.find_element_by_css_selector('#content > div:nth-child(2) > ul > li > div.info_area > div.subject > span > a').text)
        # if cite_name not in cite_names:
        #     cite_names.append(cite_name)

        no_space_al = str(al).replace(' ', '') # remove empty spaces in names
        if no_space_al not in page_img_text and page_img_text not in no_space_al:
            print(f'{al} is not on the page.{no_space_al} != {page_img_text}')
        else:
        #searchbox.submit()
            cnt+=1
        searchbox = driver.find_element_by_name('query').clear()
    except(NoSuchElementException):
        searchbox = driver.find_element_by_name('query').clear()
        print(f'{al} info is not found.')

print(cnt)
#print(cite_names)