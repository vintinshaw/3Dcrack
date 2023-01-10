# -*- coding: utf-8 -*-
from scipy import signal
wind_width1 = 5
wind_width2 = 150
win1 = signal.windows.gaussian(wind_width1, std=wind_width1)
win2 = signal.windows.gaussian(wind_width2, std=wind_width2)
sum_w1 = sum(win1)
sum_w2 = sum(win2)
import os
import cv2
import numpy as np

def mynorm(img):
    img = np.pad(img, wind_width2 // 2, mode="reflect")
    res = np.zeros_like(img)
    for index, line in enumerate(img):
        filtered1 = np.convolve(line, win1, mode='same') / sum_w1
        filtered2 = np.convolve(filtered1, win2, mode='same') / sum_w2
        res[index] = filtered1 - filtered2
    return res[wind_width2 // 2: -wind_width2 // 2,wind_width2 // 2:-wind_width2 // 2]

# .ran
def load3DData(str_3d_data_path):
    import os

    with open(str_3d_data_path, 'rb') as binFile:

        n_3dData_w = int(np.fromfile(binFile, '1i', 1)[0])
        n_3dData_h = int(np.fromfile(binFile, '1i', 1)[0])
        if os.path.getsize(str_3d_data_path) > 10485768:
            _ = np.fromfile(binFile,'1i',1)
        np_s_3dData = np.fromfile(binFile, '1h', n_3dData_h * n_3dData_w).reshape(n_3dData_h, n_3dData_w).astype(np.float32)
        binFile.close()
    np_s_3dData = mynorm(np_s_3dData)
    return np_s_3dData
def main():
    savedir='/home/vintinshaw/3Dcrack/data/RAND_CRACK/unlabeled_tif'
    
    path='/media/dataRep1/RawSource/2022detectdata/铜仁3D/12750_2022_03_08_10_49_14_GMS_01_001_10034_SICK_3D/camera/raw3D'
    
    path='/media/dataRep1/RawSource/2022detectdata/铜仁3D/12754_2022_03_08_11_52_46_GMS_01_001_10036_SICK_3D/camera/raw3D'
    
    path='/media/dataRep1/RawSource/2022detectdata/铜仁3D/12750_2022_03_08_10_49_14_GMS_01_001_10036_SICK_3D/camera/raw3D'
    
    path='/media/dataRep1/RawSource/2022detectdata/铜仁3D/12755_2022_03_08_14_15_07_GMS_01_001_10034_SICK_3D/camera/raw3D'
    
    path='/media/dataRep1/RawSource/2022detectdata/铜仁3D/12751_2022_03_08_11_39_03_GMS_01_001_10034_SICK_3D/camera/raw3D'
    
    path='/media/dataRep1/RawSource/2022detectdata/铜仁3D/12755_2022_03_08_14_15_07_GMS_01_001_10036_SICK_3D/camera/raw3D'
    
    path='/media/dataRep1/RawSource/2022detectdata/铜仁3D/12751_2022_03_08_11_39_03_GMS_01_001_10036_SICK_3D/camera/raw3D'
    
    path='/media/dataRep1/RawSource/2022detectdata/铜仁3D/12757_2022_03_08_14_56_29_GMS_01_001_10034_SICK_3D/camera/raw3D'
    pre=path.split('/')[6]
    print(pre)
    filelist=os.listdir(path)
    print(len(filelist))
    for file in filelist:
        if(file.endswith('.ran')):
            filename=os.path.join(path,file)
            np_s_3dData=load3DData(filename)
            cv2.imwrite(os.path.join(savedir,pre+file[:-4]+'.tif'),np_s_3dData)
if __name__ == "__main__":
    main()