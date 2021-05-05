from PIL import Image
import os, sys


pos_list = [2, 3, 4, 6, 7, 8, 9, 10]
for pos_val in pos_list:
    path = os.path.join('tv_human_interactions_videos', f'frames_pos_{pos_val}')
    i = 0
    for subpath in os.listdir(path):
        for item in os.listdir(path+"/"+subpath):
            im = Image.open(path+"/"+subpath+"/"+item)
            imResize = im.resize((300,300), Image.ANTIALIAS)
            components = item.split("_")
            imResize.save(path+"/"+subpath+"/"+str(i)+"_"+components[1]+"_"+components[2] , 'PNG')
            os.remove(path+"/"+subpath+"/"+item)
        i = i+1
