import tensorflow as tf
from PIL import Image
import os, glob

from keras.applications.resnet import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.layers import Flatten, Dense, BatchNormalization, Activation
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

dir = "C:/Users/Soohyun/Desktop/PythonProject/multicamPy/프로젝트/술이미지크롤링/우리술분류/우리술이미지_png"
alcohol_categories = os.listdir(dir)
total_categorie_length = len(alcohol_categories)

image_width = 64
image_height = 64

X = []
Y = []
for idx, cat in enumerate(alcohol_categories):
    
    # label  
    label = [0 for i in range(total_categorie_length)]
    label[idx] = 1

    # image  
    image_dir = dir + "/" + cat
    print(image_dir)
    files = glob.glob(image_dir+"/*.png")
    for i, f in enumerate(files):
        img = Image.open(f) 
        img = img.convert("RGB")
        img = img.resize((image_width, image_height))

        # change as numpy arr
        data = np.asarray(img)      
        X.append(data)
        Y.append(label)


X = np.array(X)
Y = np.array(Y)

# check shape 
print("Check: " , X.shape, Y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, Y)
xy = (X_train, X_test, y_train, y_test)

print("npy로 저장")
np.save("C:/Users/Soohyun/Desktop/PythonProject/multicamPy/프로젝트/술이미지크롤링/우리술분류/image", xy)
print("저장완료: ", len(Y))


# categorize
lego_categories = os.listdir(dir)
total_categorie_length = len(lego_categories)

# set image size
image_w = 64
image_h = 64

# load data
X_train, X_test, y_train, y_test = np.load("C:/Users/Soohyun/Desktop/PythonProject/multicamPy/프로젝트/술이미지크롤링/우리술분류/image.npy", allow_pickle=True)


# normalization (0~1)
X_train = X_train.astype("float") / 256
X_test  = X_test.astype("float")  / 256
print('X_train shape:', X_train.shape)

# model structure 
dropout = 0.25


resnet50 = ResNet50(weights = 'imagenet', input_shape = (64, 64, 3), include_top = False)
#vgg19 = VGG19(weights = 'imagenet', input_shape = (64, 64, 3), include_top = False)

model = tf.keras.models.Sequential()
model.add(resnet50)
model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(tf.keras.layers.Dropout(dropout))
model.add(Dense(total_categorie_length, activation = 'softmax'))

 

# build model
model.compile(loss='categorical_crossentropy',   # optimize function
    optimizer='rmsprop',
    metrics=['accuracy'])

print(model.summary())


# save model as hdf5
hdf5_file = "C:/Users/Soohyun/Desktop/PythonProject/multicamPy/프로젝트/술이미지크롤링/우리술분류/4obj-model.hdf5"
if os.path.exists(hdf5_file):
    # load trained model 
    model.load_weights(hdf5_file)
else:
    # save result if trained model not exists
    model.fit(X_train, y_train, batch_size=32, epochs=50)
    model.save_weights(hdf5_file)

# eval 
score = model.evaluate(X_test, y_test)
print('loss=', score[0])        # loss
print('accuracy=', score[1])    # acc


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
df = pd.read_csv('C:/Users/Soohyun/Desktop/AI 프로젝트 1조/전통주정보_크롤링.csv', encoding='cp949')
#print(df.head())

test_img = "C:/Users/Soohyun/Desktop/PythonProject/multicamPy/프로젝트/술이미지크롤링/우리술분류/오매백주_이미지.jpg"
img = Image.open(test_img)
img = img.convert("RGB")
img = img.resize((64, 64))
data = np.asarray(img)
x = np.array(data)
x = x.astype("float") / 256 
x = x.reshape(-1, 64, 64, 3)
prediction = model.predict(x)
result = [np.argmax(value) for value in prediction]
술_예측값 = str(lego_categories[result[0]])


print("")
print("술 이름: ", 술_예측값)

네이버지식백과_술리스트 = df['상품명'].tolist()
print(네이버지식백과_술리스트)
술정보_컬럼 = df['상품명'] == 술_예측값
술정보 = df[술정보_컬럼]
print(술정보)

