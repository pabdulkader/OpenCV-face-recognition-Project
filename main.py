import cv2
import os
import numpy as np

#mapping names to numbers
def name_map(x):
    if x=='sachin':
        return 0
    if x=='obama':
        return 1
    if x=='messi':
        return 2
    
#mapping numbers to name
def get_name(x):
    if x==0:
        return 'sachin'
    if x==1:
        return 'obama'
    if x==2:
        return 'messi'
    
def detect_face(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face_cascade=cv2.CascadeClassifier(r'C:\Users\HP\OneDrive\Desktop\deep_learning\face_training\haarcascade_frontalface_default.xml')
    faces=face_cascade.detectMultiScale(gray)
    
    if(len(faces)==0):
        return None,None
    
    (x,y,w,h)=faces[0]
    
    return gray[y:y+w,x:x+h],faces[0]


def prepare_training_data(data_folder_path):
    dirs=os.listdir(data_folder_path)
    
    faces=[]
    labels=[]
    
    for dir_name in dirs:
        
        label=dir_name
        
        subject_dir_path=data_folder_path+'\\'+dir_name
        
        subject_img_names=os.listdir(subject_dir_path)
        
        for img_name in subject_img_names:
            img_path=subject_dir_path+'\\'+img_name
            
            image=cv2.imread(img_path)
            
            cv2.imshow('training images.....',image)
            cv2.waitKey(1000)
            
            #detecting face
            face,rect=detect_face(image)
            if face is not None:
                
                faces.append(face)
                labels.append(name_map(label))
                print(labels)
                
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    return faces,labels

print('prepare data.....')
faces,labels=prepare_training_data(r'C:\Users\HP\OneDrive\Desktop\deep_learning\face_training\DB')
print('data prepared')


face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces,np.array(labels))

def draw_rectangle(img,rect):
    (x,y,w,h)=rect
    cv2.rectangle(img,pt1=(x,y),pt2=(x+w,y+h),color=(0,255,0),thickness=2)
    
def draw_text(img,text,x,y):
    cv2.putText(img,text,(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

def predict(test_img):
    
    img=test_img.copy()
    
    #detect face
    face,rect=detect_face(img)
    
    #predict
    label=face_recognizer.predict(face)
    
    #find name
    lable_text=get_name(label[0])
    
    #rect,text
    draw_rectangle(img,rect)
    draw_text(img,lable_text,rect[0],rect[1])
    
    return img

imgg=input("image")
# test_img1=cv2.imread(r'C:\Users\HP\OneDrive\Desktop\deep_learning\face_training\DB\messi\02.jpg')
test_img1=cv2.imread(imgg)
predicted=predict(test_img1)
cv2.imshow('predicted image',predicted)
cv2.waitKey(0)
cv2.destroyAllWindows()