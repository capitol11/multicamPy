import os
import cv2
import numpy as np

# load YOLO
net = cv2.dnn.readNet("./YOLO/yolov3.weights", "./YOLO/yolov3.cfg")
classes = []
with open("./YOLO/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

# path = './agumentation/'
# source_img_name = 'img.png'
# target_img_name = 'cutter.png'


### png인지 파일명 확장자랑 path 명칭 정확히 맞지 않으면 생성 안됨.
path = './img_png/명작 복분자'
source_img_name = "명작 복분자_1.png"
target_img_name = '커터.png'


# load img
def slice_image_and_save(path, source_img_name, target_img_name):

    # np.fromfile to solve Korean path recognization problem in cv2
    image_path = path + '/' + source_img_name
    img_temp = np.fromfile(image_path, np.uint8)
    img = cv2.imdecode(img_temp, cv2.IMREAD_COLOR)

    try:
    #img = cv2.imread(image_path)
        img = cv2.resize(img, None, fx=0.3, fy=0.3)
        height, width, channels = img.shape
    except Exception as e:
        print(str(e))

    # load img as blob
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # detect alcohol bottle
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes: x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[i]


        if label == "bottle":
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            # cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
            print("술")
            print(f'{x}, {y}, {w}, {h}')
            cutter = img[y:y + h, x:x + w].copy()

            ext = os.path.splitext(target_img_name)[1]   # 확장자만 가져오기
            result, n = cv2.imencode(ext, cutter, params=None) # cutter 인코딩

            if result:
                with open(target_img_name, mode='w+b') as f:
                    n.tofile(f)

            cv2.imwrite('./agumentation/' + target_img_name, cutter)
            cv2.imshow("Image", img)
            cv2.imshow('sliced image', cutter)
            print('저장경로: ', path + '/' +target_img_name)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break

'''
The image file with {source_img_name} will be loaded in {path} and sliced partly.
After that the result will be saved in a same path with {target_img_name}
'''

#new_path = 'C:/Users/Soohyun/Desktop/술/걍즐겨'
#source_img_name = '걍즐겨_1.jpg'
#dest_img_name = '컷_걍즐겨.jpg'

#os.chdir(new_path)
#files = os.listdir(new_path)
#print(files)

print("Slicing images")
#for index, al in enumerate(files):
#    slice_image_and_save(new_path, al, dest_img_name[:-4]+'_'+str(index)+'.jpg')
#    print(dest_img_name[:-4]+"_"+str(index)+'.jpg')
print("Done.")

slice_image_and_save(path, source_img_name, target_img_name)