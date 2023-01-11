import cv2
# path='/home/vintinshaw/3Dcrack/data/RAND_CRACK/JPEGImages/genID-00000036-1_01162a.tif'
path='/home/vintinshaw/3Dcrack/data/RAND_CRACK/unlabeled_tif/1_00217.tif'

img=cv2.imread(path,cv2.IMREAD_ANYDEPTH)
print(img)
print(img.max())
print(img.min())

print(img.shape)