from mayavi import mlab
# import cv2
#
#
# def main():
#     # path='/home/vintinshaw/3Dcrack/data/RAND_CRACK/JPEGImages/genID-00000036-1_01162a.tif'
#     path = 'dataPreview/12750_2022_03_08_10_49_14_GMS_01_001_10034_SICK_3D1_01015.tif'
#
#     img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
#     mlab.surf(img)
#     mlab.show()
#     mlab.clf()
#
#
# if __name__ == '__main__':
#     main()

# from numpy import pi, sin, cos, mgrid
# dphi, dtheta = pi/250.0, pi/250.0
# [phi,theta] = mgrid[0:pi+dphi*1.5:dphi,0:2*pi+dtheta*1.5:dtheta]
# m0 = 4; m1 = 3; m2 = 2; m3 = 3; m4 = 6; m5 = 2; m6 = 6; m7 = 4;
# r = sin(m0*phi)**m1 + cos(m2*phi)**m3 + sin(m4*theta)**m5 + cos(m6*theta)**m7
# x = r*sin(phi)*cos(theta)
# y = r*cos(phi)
# z = r*sin(phi)*sin(theta)
#
# # View it.
# from mayavi import mlab
# s = mlab.mesh(x, y, z)
# mlab.show()