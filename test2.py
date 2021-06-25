from PIL import Image
import pandas as pd
from IPython.display import display
import pprint

fileName=["pic1", "pic2", "pic3"]
dataset = []

for i in fileName:
    tmp = Image.open('./picture/' + i +'.jpg')
    fileName=i+'.jpg'
    fm = tmp.format
    fw = tmp.width
    fh = tmp.height
    print(fileName, fw, fh, fm)
    dataset.append([fileName, fw, fh, fm])
    tmpResize = tmp.resize((300,300))
    tmpResize.save('./picture/'+i+'_300.jpg')
print(dataset)

df = pd.DataFrame(dataset, columns=["file", "width", "height", "format"])
print(df)

df.to_csv('./파일정보.csv')
display(df) #IPython
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(df)
