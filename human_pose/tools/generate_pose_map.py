import numpy as np
import pandas as pd
import os
import json
from tqdm import tqdm


MISSING_VALUE=-1

#将坐标从字符串加载成list的数据
def load_pose_cords_from_strings(y_str,x_str):
    y_cords=json.loads(y_str)
    x_cords=json.loads(x_str)
    return np.concatenate([y_cords[:,None],x_cords[:,None]],axis=1)

#根据关键点坐标来生成多通道热力图
def cords_to_map(cords,img_size,sigma=6):
    '''
        cords:传入18*2的矩阵,其中记录关键点的坐标，
        img_size:表示原始图片的大小，
        sigma:根据距离来计算相应值时会用一个参数
    '''
    result=np.zeros(img_size+cords.shape[0],dtype='uint8')

    for i,point in enumerate(cords):
        if point[0] == MISSING_VALUE or point[1]== MISSING_VALUE:
            continue
        
        #这里图片大小是256*178，h*w,w对应的是x
        xx,yy=np.meshgrid(np.arange(result.shape[1]),np.arange(result.shape[0]))
        #在每张通道图上根据与该图关键位置的远近来赋予像素值
        result[...,i]=np.exp(-((yy-point[0])**2+(xx-point[1])**2)/(2*sigma**2))
    return result

#根据调用上面两个函数，生成每张图片的相应的18通道的矩阵，并将矩阵保存为np文件
def compute_pose(img_size:tuple,keypoint_csv,savePath,sigma):
    keypoint_csv=pd.read_csv(keypoint_csv,sep=':')
    keypoint_csv=keypoint_csv.set_index('name')    #使用已经存在的列来作为index

    for i in tqdm(range(len(keypoint_csv))):
        row=keypoint_csv.iloc[i]
        name=row.name
        print(savePath,name)
        file_name=os.path.join(savePath,name+'.npy')
        kp_array=load_pose_cords_from_strings(row.keypoints_y,row.keypoints_x)


        pose=cords_to_map(kp_array,img_size,sigma)
        np.save(file_name,pose)

img_size=(256,178)   #图像的高宽，根据需要自行更换
keypoint_csv=''
savePath=''
sigma=6

compute_pose(img_size,keypoint_csv,savePath,sigma)

    

