from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import os
import urllib.request


def createFolder(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except OSError:
        print('Error: directory is already created. ' + dir)


def check_img_format(url):
    if "jpeg" or "jpg" or "JPG" in url:
        return ".jpg"
    elif "png" or "PNG":
        return ".png"


keyword = "hund"
createFolder(f'./{keyword}/_img_download')

driver = webdriver.Chrome('c:/pydata/chromedriver.exe')
driver.implicitly_wait(2)

base_url = 'https://www.google.de/?&bih=937&biw=1254&hl=de'
driver.get(base_url)
print(keyword, 'ist gefunden.')


input_value = driver.find_element_by_xpath('/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/div/div[2]/input')
input_value.send_keys(keyword)
#driver.find_element_by_xpath('/html/body/div[1]/div[3]/form/div[1]/div[1]/div[3]/center/input[1]').click()
input_value.send_keys(Keys.RETURN)
driver.find_element_by_xpath('//*[@id="hdtb-msb"]/div[1]/div/div[2]/a').click() # click Bilder
#driver.back() # go to previous page
#class_lst = driver.find_elements_by_class_name('bRMDJf.islir')
class_lst = driver.find_element_by_class_name('bRMDJf.islir')
#for i in class_lst:
#    print(i.find_element_by_tag_name('img').get_attribute('src'))
img_url = str(class_lst.find_element_by_tag_name('img').get_attribute('src'))
# img_url = img_url[img_url.rfind(',')+1:]  # save string after ,
print(img_url)

img_format = check_img_format(img_url)
savename = './hund/_img_download/HUND' + img_format

urllib.request.urlretrieve(img_url, savename)
print("picture is saved.")

