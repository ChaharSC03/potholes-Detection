import cv2
import numpy as np 
import os
import yaml
from yaml.loader import SafeLoader

class YOLO_Pred():
    def __init__(self,onnx_model,data_yaml):
        with open(data_yaml, mode='r') as f:
            data_yaml=yaml.load(f,Loader=SafeLoader)

        self.labels=data_yaml['names']
        self.nc=data_yaml['nc']


        self.yolo =cv2.dnn.readNetFromONNX(onnx_model)
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU) 


    def prediction_img(self, image):
        row, col, d= image.shape
        max_rc=max(row, col)
        input_img=np.zeros((max_rc, max_rc,3), dtype=np.uint8)

        input_img[0:row, 0:col]=image
        input_width_yolo=640
        input_heigth_yolo=640
        blob=cv2.dnn.blobFromImage(input_img,1/255,(input_width_yolo, input_heigth_yolo),swapRB=True, crop=False)
        self.yolo.setInput(blob)
        prediction=self.yolo.forward()

        detection=prediction[0]
        boxes=[]
        confidences=[]
        calsses=[]

        img_w, img_h= input_img.shape[:2]
        xfactor=img_w/input_width_yolo
        yfactor=img_h/input_heigth_yolo

        for i in range(len(detection)):
            row=detection[i]
            confidence=row[4]
            if confidence>0.4:
                class_score=row[5:].max()
                class_id=row[5:].argmax()

                if class_score >0.25:
                    cx,cy,w,h=row[0:4]
                    left=int((cx-0.5*w)*xfactor)
                    top=int((cy-0.5*h)*yfactor)
                    width=int(w*xfactor)
                    height=int(h*yfactor)

                    box=np.array([left, top, width, height])

                    confidences.append(confidence)
                    boxes.append(box)
                    calsses.append(class_id)

        confidences_np=np.array(confidences).tolist()
        boxes_np=np.array(boxes).tolist()

        index= cv2.dnn.NMSBoxes(boxes_np,confidences_np, 0.25, 0.45)

        for ind in index:
            x,y,w,h=boxes_np[ind]
            bb_conf=int(confidences_np[ind]*100)
            class_id=calsses[ind]
            class_name=self.labels[class_id]
            colors=self.generate_colors(class_id)

            text=f'{class_name}:{bb_conf}%'

            cv2.rectangle(image, (x,y),(x+w,y+h),colors,2)
            cv2.rectangle(image, (x,y-30),(x+w,y),colors,-1)

            cv2.putText(image, text,(x,y-10), cv2.FONT_HERSHEY_PLAIN,0.7,(0,0,0),1)

        return image
        
    def generate_colors(self,ID):
        np.random.seed(10)
        colors=np.random.randint(100,255,size=(self.nc,3)).tolist()
        return tuple(colors[ID])

        

            
        