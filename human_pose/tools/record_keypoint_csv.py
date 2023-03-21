import numpy as np
import os
import pandas as pd
from ..utils.body import Body
from tqdm import tqdm
import cv2

#每张图片只记录可信度最高的那个的关键点信息

def get_single_keypoints_cordinates(subset,candidate):

    if len(subset)!=0:
        cordinates=[]

        single_index=np.argmax(subset[:,-2])

        for part in subset[single_index,:-2]:
            if part==-1:
                cordinates.append([-1,-1])
            else:
                Y=candidate[part.astype(int),0]
                X=candidate[part.astype(int),0]
                cordinates.append([X,Y])
    else:
        cordinates=[[-1,-1]*18]
    return np.array(cordinates).astype(int)


def get_mult_keypoints_cordinates(subset:np.ndarray,candidate:list,num_max:int=1)->np.ndarray:
    '''
        将subset和candidate中记录的信息整合起来,得到关键点和坐标的直接对应关系
        subset:n*20,其中n表示图片中检测到人体数目,前18列时关键点和峰值点序号的对应关系,
                倒数第二列是该人体的得分，得分越高表示该人体存在的概率越高，最后一列表示该人体检测到的关键点数目
        candidate: list ,list中按照峰值点的编号存放着相应的峰值点信息(y,x,像素值,峰值点编号)
        num_max: 表示最多取几个人体，按照检测到的人体得分进行排序，取得分高的前num_max的人体
    
        return ndarray num*18*2的矩阵
    '''
    multi_cordinates=[]
    if len(subset)!=0:
        
        subset=subset[np.argsort(subset[:,-2]),:]
        # single_index=np.argmax(subset[:,-2])
        #取前n列
        subset=subset[:num_max]

        for i in range(num_max):
            cordinates=[]
            for part in subset[i,:-2]:
                if part==-1:
                    cordinates.append([-1,-1])
                else:
                    Y=candidate[part.astype(int),0]
                    X=candidate[part.astype(int),0]
                    cordinates.append([X,Y])
            multi_cordinates.append(cordinates)

    else:
        cordinates=[[-1,-1]*18]
        multi_cordinates.append(cordinates)
    return np.array(multi_cordinates).astype(int)


def record_keypoint_csv(input_folder,output_path,model):
    if os.path.exists(output_path):
        processed_names=set(pd.read_csv(output_path,sep=':')['name'])  #构建已有名称的集合
        result_file=open(output_path,'a')
    else:
        result_file=open(output_path,'w')
        processed_names=set()
        result_file.write('name:keypoints_y:keypoints_x\n')

    for img_name in tqdm(os.listdir(input_folder)):
        if img_name in processed_names:
            continue

        oriImg=cv2.imread(os.path.join(input_folder,img_name))
        candidate, subset = model(oriImg)
        
        #这里我们同样只记录得分最高人的人体关键点
        pose_cords=get_mult_keypoints_cordinates(subset,candidate,1).squeeze(0)
        result_file.write(f'{img_name}:{str(pose_cords[:,0])},{str(pose_cords[:,1])}\n')

        result_file.flush()   #数据立即写入文件，不放入缓冲区


input_folder=''                            #待获取人体关键点的文件夹
output_path=''                              #存放结果csv文件的路径

body_estimation = Body('model/body_pose_model.pth')

record_keypoint_csv(input_folder,output_path,body_estimation)