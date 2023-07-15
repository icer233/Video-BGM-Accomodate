import os
import cv2
import shutil
import random
from PIL import Image
import numpy as np
from keras.models import load_model
from ctypes.wintypes import HACCEL
from opcode import hascompare
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
video_path=input()
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
timef=10
tmp=0
try:
    os.system("rm -rf Music_Materials")
except:
    tmp=0
face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
classifier=load_model('./model_v_47.hdf5')
class_labels={0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
audios=['Killing In The Name','The Last Puzzle','Paris','Curly Wurly','仲夏夜','你不要难过','aLIEz ','Raining Blood','The Last Puzzle','Larkin-Mantis Lords','久石譲 - 風のとおり道','秋水长','穿越时空的思念','Counting Stars','Angel Of Death','The Last Puzzle','Belladonna','Funkytown','Canon in D Major','错位时空','Enemy']
def Emotion(img):    
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    try:
        (x,y,w,h)=faces[0]
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        face=roi_gray
        roi = face.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        preds = classifier.predict(roi)[0]
        label = class_labels[preds.argmax()]
        return label
    except:
        a=0
def GetMusic():
    pre=0
    pre_emotion=-1
    pro_emotion=-1
    very_emotion=-1
    for i in range(1,imageNum+1):
        result=Emotion(cv2.imread("./Tmp/"+str(i)+'.jpg',1))
        if result==pre_emotion and pre_emotion==pro_emotion and very_emotion!=result:
            very_emotion=result
            values=list(class_labels.values())
            index=0
            for j in values:
                if result==j:
                    index+=random.randint(0,2)*7
                    print(f"The person in the video is {result}, suggest using music {audios[int(index)]} from {round(timef/fps*pre,1)}s to {round(timef/fps*i,1)}s\n")
                    pre=i
                    shutil.copy('./Audios/'+audios[int(index)]+'.mp3','./Music_Materials/')
                    break
                index+=1
        elif i==imageNum:
            values=list(class_labels.values())
            index=0
            for j in values:
                if result==j:
                    index+=random.randint(0,2)*7
                    print(f"The person in the video is {result}, suggest using music {audios[int(index)]} from {round(timef/fps*pre,1)}s to {round(timef/fps*i,1)}s\n")
                    pre=i
                    shutil.copy('./Audios/'+audios[int(index)]+'.mp3','./Music_Materials/')
                    break
                index+=1
            break
        pro_emotion=pre_emotion
        pre_emotion=result
    print("All listed music files have been placed in the folder ""Music_Materials"".")
try:
    os.mkdir('./Tmp')
except:
    tmp=0
isOpened = cap.isOpened
sum=0
imageNum=0
while (isOpened):
    sum+=1
    (frameState, frame) = cap.read()
    if frameState == True and (sum % timef==0):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(np.uint8(frame))
        frame = np.array(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        imageNum = imageNum + 1
        fileName = './Tmp/' + str(imageNum) + '.jpg'
        cv2.imwrite(fileName, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
    elif frameState == False:
        break
cap.release()
print(imageNum)
try:
    os.mkdir('./Music_Materials')
except:
    tmp=0
GetMusic()
try:
    os.system("rm -rf Tmp")
except:
    os.system("del Tmp")
