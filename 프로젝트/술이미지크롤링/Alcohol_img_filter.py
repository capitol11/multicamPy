import os
import cv2

from 프로젝트.술이미지크롤링.OpenCV_detection import slice_image_and_save

path = 'C:/Users/Soohyun/Desktop/술'
os.chdir(path) # 해당 폴더로 이동
files = os.listdir(path)

#for index, alcohol in enumerate(files):
    #total_path = path+'/'+alcohol
    #alcohol_lst = os.listdir(total_path)
    #destination_path = './argumentation/' + alcohol
    #file_name = f'alcohol{index}'
slice_image_and_save("./agumentation/img.png", './agumentation/', 'cutter')


