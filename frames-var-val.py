import cv2
import numpy as np
import matplotlib.pyplot as plt
import os,glob

pos_list = list(range(2, 11))
os.chdir("tv_human_interactions_videos")

for pos_val in pos_list:
    print(f'Start processing value {pos_val}')
    frame_dir = 'frames_pos_' + str(pos_val)
    if not os.path.isdir(frame_dir):
        os.mkdir(frame_dir)
    for file in glob.glob("*.avi"):
        videoFile = os.path.basename(file)

        imagesFolder = videoFile.split(".")[0]

        cap = cv2.VideoCapture(videoFile)
        pos = 0
        i = 0
        
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        while(cap.isOpened()):
            ret, frame = cap.read()
            class_folder = os.path.join(frame_dir, imagesFolder.split("_")[0])
            frame_name = imagesFolder+"_"+str(i)+'.png'
            if not os.path.isdir(class_folder):
                os.mkdir(class_folder)
            f1 = os.path.join(class_folder, frame_name)
            if ret:
                s = cv2.imwrite(f1,frame)
                if (s == False):
                    print(f'Writing frame {pos} of video {videoFile} failed!')
            i = i + 1
            pos = pos + pos_val
            cap.set(cv2.CAP_PROP_POS_FRAMES,pos)
            if (pos >= length):
                break
            #prvs = next
        cap.release()
        cv2.destroyAllWindows()
    print(f'Successfully processing value {pos_val}')



