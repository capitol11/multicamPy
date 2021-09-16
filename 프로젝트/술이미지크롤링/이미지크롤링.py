from pydantic.datetime_parse import time
from selenium import webdriver
from bs4 import BeautifulSoup as bs
import time


def get_base_url():
    return current_url

class Base_URL:
    global current_url

    driver = webdriver.Chrome(r'C:/pydata/chromedriver')
    url = 'http://www.naver.com'

    driver.get(url)
    time.sleep(1) # 페이지 열리는거 잠시 기다림

    html_source = driver.page_source
    soup = bs(html_source, "html.parser")

    driver.find_element_by_id("query").click()
    element = driver.find_element_by_id("query")
    element.send_keys("까메오 막걸리")

    # 실행
    driver.find_element_by_id("search_btn").click()
    driver.find_element_by_xpath('//*[@id="lnb"]/div[1]/div/ul/li[3]/a').click()
    current_url = driver.current_url
    driver.close()




