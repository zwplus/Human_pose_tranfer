import cv2
import numpy as np
import math
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import os
import util
from model import bodypose_model

class Body(object):


    def __init__(self, model_path):
        '''
            初始化模型模型，并加载模型权重
            模型模型模式为评估模式
        '''
        self.model = bodypose_model()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        model_dict = util.transfer(self.model, torch.load(model_path))
        self.model.load_state_dict(model_dict)
        self.model.eval()

    def __call__(self, oriImg):
        scale_search = [0.5, 1.0, 1.5, 2.0]     #在不同尺度上检测人体姿态，这里表示放缩比例
        # scale_search = [0.5]
        boxsize = 368
        stride = 8    
        padValue = 128   #填充的边界值
        
        #判断关键点和肢体是否存在的阈值
        thre1 = 0.1      
        thre2 = 0.05
        
        multiplier = [x * boxsize / oriImg.shape[0] for x in scale_search]
        
        heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))    #19通道的热力图
        paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))        #记录人体各个肢体方向的图2*19，其中两个肢体不包括，一个方向由两个通道构成，一个表示x,一个表示y

        for m in range(len(multiplier)):   #在不同尺度上找峰值点和相应的边，然后最后统一缩放回原始图的大小，求平均，得到最终的峰值点图和肢体图
            scale = multiplier[m]
            
            #将将图片变成和x*boxsize一样大小
            imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            #不满足滑动窗口整除时，添加padding，使得其满足滑动窗口
            imageToTest_padded, pad = util.padRightDownCorner(imageToTest, stride, padValue)
            
            #先增加一个batch维度，然后调整通道顺序，变成BCHW
            im = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 256 - 0.5
            im = np.ascontiguousarray(im)    #将im从存储上的不连续转换成连续值，提高运算效率

            #输入到模型中去
            data = torch.from_numpy(im).float()
            if torch.cuda.is_available():
                data = data.cuda()
            # data = data.permute([2, 0, 1]).unsqueeze(0).float()
            with torch.no_grad():
                Mconv7_stage6_L1, Mconv7_stage6_L2 = self.model(data)     #分别代表关键点的检测结果和肢体的检测结果
            Mconv7_stage6_L1 = Mconv7_stage6_L1.cpu().numpy()
            Mconv7_stage6_L2 = Mconv7_stage6_L2.cpu().numpy()

            # extract outputs, resize, and remove padding
            # heatmap = np.transpose(np.squeeze(net.blobs[output_blobs.keys()[1]].data), (1, 2, 0))  # output 1 is heatmaps
            #关键点应该是batch*19*h*w,去掉通道维，将图片调整成BHWC
            heatmap = np.transpose(np.squeeze(Mconv7_stage6_L2), (1, 2, 0))  # output 1 is heatmaps
            #因为stride，所以大小被缩小，这里放大回原始输入的大小
            heatmap = cv2.resize(heatmap, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
            #除去原始图片的padding
            heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
            #放缩回原始图片的大小
            heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
            
            #paf 代表每个像素点所在肢体的方向向量 part affinity fields,所以应该是肢体的数目乘2
            # paf = np.transpose(np.squeeze(net.blobs[output_blobs.keys()[0]].data), (1, 2, 0))  # output 0 is PAFs
            paf = np.transpose(np.squeeze(Mconv7_stage6_L1), (1, 2, 0))  # output 0 is PAFs
            paf = cv2.resize(paf, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
            paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
            paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

            heatmap_avg += heatmap_avg + heatmap / len(multiplier)     #多尺度值取平均
            paf_avg += + paf / len(multiplier)                  #paf也是多尺度的结果取平均

        
        #这里找到每个通道的峰值点，峰值的点的要求其要大于上下左右4个像素点，并且要高于阈值
        #将找到的峰值点记录到all_peaks中，all_peaks中有18个list 一个list表示一个关键点对应的通道图中的待选峰值点，峰值点表示如下(y,x,像素值,编号)
        all_peaks = []
        peak_counter = 0

        for part in range(18):
            map_ori = heatmap_avg[:, :, part]
            
            #高斯滤波，除去一些噪声
            one_heatmap = gaussian_filter(map_ori, sigma=3)

            map_left = np.zeros(one_heatmap.shape)
            map_left[1:, :] = one_heatmap[:-1, :]
            map_right = np.zeros(one_heatmap.shape)
            map_right[:-1, :] = one_heatmap[1:, :]
            map_up = np.zeros(one_heatmap.shape)
            map_up[:, 1:] = one_heatmap[:, :-1]
            map_down = np.zeros(one_heatmap.shape)
            map_down[:, :-1] = one_heatmap[:, 1:]
            
            #峰值的二值图像，
            #利用阈值thre1和left,right找到峰值点
            #np.logical_and求两个矩阵的逐个元素的与
            #因为这里由多个元素，所以这里增加reduce来处理多元素问题
            #np.ufunc.reduce(array,aixs=0),这里reduce是减小指定维度的作用，如这里array里是一个个二维矩阵，所以先将array变成一个三维矩阵即C*H*W，然后沿着C维度逐步逐步应用ufunc，最后将C压缩到1，最后将该维度除去
            peaks_binary = np.logical_and.reduce(
                (one_heatmap >= map_left, one_heatmap >= map_right, one_heatmap >= map_up, one_heatmap >= map_down, one_heatmap > thre1))
            
            #np.nonzero记录着一系列非0元素的索引值，其结果是一个元组，元组里有2个array,一个array里全x轴坐标，一个array里全y轴坐标
            #peaks_binary  peaks里有很多元组，一个元组就是一堆坐标(y,x)
            peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
            #peaks_with_score peaks_with_score 里的元组含有3个值(y,x,score)，score的值是map_ori对应点的像素值 
            peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
            
            #峰值点计数，计数结果作为峰值点的id
            peak_id = range(peak_counter, peak_counter + len(peaks))
            
            #将峰值id加到峰值的相关信息里
            peaks_with_score_and_id = [peaks_with_score[i] + (peak_id[i],) for i in range(len(peak_id))]

            all_peaks.append(peaks_with_score_and_id)
            peak_counter += len(peaks)

        #起点存在相同的，终点不存在相同，6到18实际上是没有的，因为关键点就是0-17，6到18就是5到17，和3到17（即2到16）实际上应该是不存在的
        limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
                [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
                [1, 16], [16, 18], [3, 17], [6, 18]]
        # the middle joints heatmap correpondence   
        mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22], \
                [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52], \
                [55, 56], [37, 38], [45, 46]]

        #因为一个关键点图中存在着多个峰值点，其可能是不同人的关键点，因此在考虑两个关键点是否构成某一个的肢体时，需要从两个关键点图中分别随机抽取一个点
        #然判断这两个点是否构成一个肢体，其判断方法是：在这两个关键点之间等距地选取10个点(就是在二者的连线上)，然后看看这10个关键点上记录的肢体方向和
        #关键点本身连线的方向是不是足够接近，如果足够接近则表明这两个关键点能够构成一个人体的肢体。


        connection_all = []    #connection_all里面也是19个list,其中最后两个list为空。前17个list里记录每一条肢体的组成。一个list有多个这样的元组(点1编号，点2编号，socre,i,j)，同一个list里点的编号不重复
        special_k = []
        mid_num = 10  
        
        for k in range(len(mapIdx)):
            
            #这个地方score_mid
            score_mid = paf_avg[:, :, [x - 19 for x in mapIdx[k]]]  #x-19 56-19+1=38=19*2
            
            #两个关节点,取第limbSeq[k][0] 通道里的峰值点,因为肢体是从这两个通道中关键点构成的
            candA = all_peaks[limbSeq[k][0] - 1]
            candB = all_peaks[limbSeq[k][1] - 1]
            
            #统计一下检测到的同一个关节点的数目，因为同一张图中可能有多个人物
            nA = len(candA)
            nB = len(candB)
            
            #两个关键点的序号
            indexA, indexB = limbSeq[k]
            
            
            if (nA != 0 and nB != 0):
                connection_candidate = []
                for i in range(nA):
                    for j in range(nB):
                        
                        #遍历同一种关键点的多个点，就想当有多个人的情况下，关键点有多个，如何去配对两个关键点生成正确的关节
                        vec = np.subtract(candB[j][:2], candA[i][:2])   #对应坐标位置相减，得到x,y两个方向的差值，这个差值来计算距离
                        norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                        norm = max(0.001, norm)
                        vec = np.divide(vec, norm)  #除以距离后相当于标准化向量了，实际上方向也包含在其中了
                        
                        #mid_num=10 
                        startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                            np.linspace(candA[i][1], candB[j][1], num=mid_num)))
                        
                        #从score_mid中取出上面startend中生成的每个点的方向向量，x,y组成方向向量(vec_x,vec_y)
                        vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                                        for I in range(len(startend))])
                        vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                                        for I in range(len(startend))])
                        
                        #用startend中每个点的方向向量和两个待匹配关键点的方向向量相乘得到值作为得分，得分越高说明方向越一致
                        #这里应该就是亲和分数
                        score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                        #平均得分+0.5*H/待匹配点的距离-1，如果待匹配点距离大于图像高度的一半，就要减一定的值，如果小于一般则加0
                        score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                            0.5 * oriImg.shape[0] / norm - 1, 0)
                        
                        #如果超过80%点的得分大于0.05就满足条件1
                        criterion1 = len(np.nonzero(score_midpts > thre2)[0]) > 0.8 * len(score_midpts)
                        #如果score_with_dist_prior为正就满足条件2
                        criterion2 = score_with_dist_prior > 0
                        
                        #说明两点满足关联
                        if criterion1 and criterion2:
                            #candA[i][2]这里面有像素值
                            connection_candidate.append(
                                [i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]])
                
                #做个排序，满足上面两个条件的在同一个i下可能不止一个j,这时候按照得分排序了
                connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
                connection = np.zeros((0, 5))
                for c in range(len(connection_candidate)):
                    i, j, s = connection_candidate[c][0:3]
                    #因为是循环，这里是判断i，j是不是已经和其他点关联了，如果关联了，则跳过否则两个点建立关联关系
                    #这里保正了同一个关节里，不存公用一个点作为出发点或者终点的情况，同一个肢体，不同人之间的峰值点一定不同的
                    #
                    if (i not in connection[:, 3] and j not in connection[:, 4]):
                        connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                        if (len(connection) >= min(nA, nB)):
                            break

                connection_all.append(connection)
            else:                
                special_k.append(k)    #记录那些一条都没有找到的肢体
                connection_all.append([])

        # last number in each row is the total parts number of that person
        # the second last number in each row is the score of the overall configuration
        subset = -1 * np.ones((0, 20))   #行数不固定，但是列数为20列的矩阵，前18列代表18个关键点，其中的值是对应峰值点的序号
        #候选峰值点，sublist 是一类关键点的集合(如果有多个人，就有多个关键点），item就具体到一个人的一个关键点
        #candidate=[(),(),(),()....]此时峰值点的id和序号对应上了
        candidate = np.array([item for sublist in all_peaks for item in sublist])

        #这段代码核心就是根据找到的肢体把同一个人的关建点合并到一起
        for k in range(len(mapIdx)):
            
            
            if k not in special_k:   # 如果k not in special_k，说明该k这个肢体是找到了,在special_k表明找不到该肢体
                
                #这里k保证了connection_all[k][:, 0] 和indexA始终代表同一种关键点，保证其是在对应的通道图里找对应关系
                
                #该肢体的两个关键点的峰值点序号
                partAs = connection_all[k][:, 0]   #多个人体的candA[i][3]
                partBs = connection_all[k][:, 1]   #多个人体的candB[j][3]
                
                #该肢体对应着那两个关键点的序号
                indexA, indexB = np.array(limbSeq[k]) - 1
                
                #len(connection_all[k])第k个肢体有多少个，即多少个人
                for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                    found = 0
                    subset_idx = [-1, -1]
                    
                    
                    #同一个肢体的里，不存在不同人公用一个峰值点的情况
                    #下面判定成立只存在不同肢体的情况，此时partAs[i]表示第I个人indexA关键点，
                    #简单点理解subset中j号人体的第8号关节点已经记录了
                    #indeA代表8号的关节点，则partAs也代表的是8号关节点，
                    for j in range(len(subset)):  # 1:size(subset,1):
                        if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                            subset_idx[found] = j
                            found += 1    #found最多为2
                    
                    
                    #说明只找到一个峰值点重合，此时只记录起点重合，不记录终点重合，
                    #如果起点相同，并且终点相同
                    if found == 1:
                        j = subset_idx[0]
                        if subset[j][indexB] != partBs[i]:    #为什么不能是subset[j][indexB] == partBs[i]，subset[j][indexA] ！= partAs[i]
                            #这个可能和limbSeq里肢体顺序有关，图是1指向0，而不是0指向1，所以只存在起始点已经被记录的情况
                            subset[j][indexB] = partBs[i]
                            subset[j][-1] += 1
                            subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                    elif found == 2:  # if found 2 and disjoint, merge them
                        j1, j2 = subset_idx
                        membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                        if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                            subset[j1][:-2] += (subset[j2][:-2] + 1)
                            subset[j1][-2:] += subset[j2][-2:]
                            subset[j1][-2] += connection_all[k][i][2]
                            subset = np.delete(subset, j2, 0)
                        else:  # as like found == 1
                            subset[j1][indexB] = partBs[i]
                            subset[j1][-1] += 1
                            subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                    # if find no partA in the subset, create a new subset
                    elif not found and k < 17:
                        row = -1 * np.ones(20)
                        row[indexA] = partAs[i]   #第i关键点对应的峰值点序号
                        row[indexB] = partBs[i]   #第j关键点对应的峰值点序号
                        row[-1] = 2
                        #总得分
                        row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                        subset = np.vstack([subset, row])
        # delete some rows of subset which has few parts occur
        deleteIdx = []
        for i in range(len(subset)):   #检测到的关键点数目小于4或者关键点的平均得分小于0.4，则删除该人体
            if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
                deleteIdx.append(i)
        subset = np.delete(subset, deleteIdx, axis=0)

        # subset: n*20 array, 0-17 is the index in candidate, 18 is the total score, 19 is the total parts
        # candidate: x, y, score, id
        return candidate, subset    #candidate=[(y,x,score,id),(),(),()。。。。]此时峰值点的id和序号对应上了   sub:[[20个值(18个点的id,得分，检测到的总点数],[],[]]



#绘制出人体骨骼图
if __name__ == "__main__":
    body_estimation = Body('../model/body_pose_model.pth')

    test_image = '/root/human_pose/pytorch-openpose/img/01_7_additional.jpg'
    oriImg = cv2.imread(test_image)  # B,G,R order
    candidate, subset = body_estimation(oriImg)
    if len(subset)>0:
        canvas = util.draw_bodypose(oriImg, candidate, subset)
        plt.imsave('../img/01_7_additional_pose.jpg',canvas[:, :, [2, 1, 0]])   #将BGR调整成RGB
    else:
        print('不足以构成人体')

