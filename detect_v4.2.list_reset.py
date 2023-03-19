import cv2
import ntpath
import os
import sys

sys.path.append('./openpose/build/python/openpose/Release');
os.environ['PATH']  = os.environ['PATH'] + ';' + './openpose/build/x64/Release;' +  './openpose/build/bin;'

import pyopenpose as op

import pytorch_lightning as pl
from LSTM.src.lstm import ActionClassificationLSTM as LSTM

import numpy as np
import torch
import torch.nn.functional as F


video_path = 'backup/black_1p_20221229_cut_084313-091233_MERGED.mp4'
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = int(cap.get(cv2.CAP_PROP_FPS))
fps = 15
tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
file_name = ntpath.basename(video_path)
vid_writer = cv2.VideoWriter('Detected_{}'.format(file_name), fourcc, 30, (width,height))

counter = 0
buffer_window = []
label = None
SKIP_FRAME_COUNT = 0

# OpenPose 모델을 로드
# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "./openpose/models/" 

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Process Image
datum = op.Datum()


# LSTM 모델 로드
CKPT_PATH = 'LSTM/lightning_logs/version_28/checkpoints/epoch=94-step=1139.ckpt'
lstm_classifier = LSTM.load_from_checkpoint(CKPT_PATH)
print(lstm_classifier)
lstm_classifier.eval()


LABELS = {
    0: "Object_Manipulation",
    1: "Wrench_job", 
}


idx =0
buffer_window = []

while True:
    ret, frame = cap.read()
    if ret == False:
        break
    for each in range(len(frame)): # 🎃
        frame[each][0:700] = 0  
    
    for each in range(len(frame)):
        frame[each][1100:] = 0 

    img = frame.copy()
    if(counter%(SKIP_FRAME_COUNT+1)==0):
        datum.cvInputData = img
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        img = datum.cvOutputData
        print(f'{idx}/{tot_frames}')
        idx += 1
        try: #🎃 신규 추가, 장면에 사람이 null일 경우를 반영
            features = []
            human_1 = datum.poseKeypoints[0]
            
            for i, row in enumerate(human_1):
                features.append(row[0])
                features.append(row[1])
                
            # print(features) # features에 50개의 x,y 좌표가 list로 담기게 된다.
            
            if len(buffer_window) < 4: # -----------  32->8->4->5로 변경 ----------- #
                buffer_window.append(features)
            
            else:
                model_input = torch.Tensor(np.array(buffer_window, dtype=np.float32))
                model_input = torch.unsqueeze(model_input, dim=0)
                y_pred = lstm_classifier(model_input)
                prob = F.softmax(y_pred, dim=1)
                pred_index = prob.data.max(dim=1)[1]
                buffer_window.pop(0)
                buffer_window.append(features)
                label = LABELS[pred_index.numpy()[0]]
                print('label:', pred_index.numpy()[0])
        except:
            pass

    if label is not None:
        cv2.putText(img, f'Action:{label}',
                   (int(width-400), height-50), cv2.FONT_HERSHEY_COMPLEX, 0.9, (102, 255, 255), 2)
    
    counter +=1
    vid_writer.write(img)      
        
cap.release()
vid_writer.release()

