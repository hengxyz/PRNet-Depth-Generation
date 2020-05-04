import numpy as np
import scipy.io as sio
from skimage.io import imread, imsave
import cv2
import os
import glob

from api import PRN
import utils.depth_image as DepthImage

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

prn = PRN(is_dlib = True, is_opencv = False) 

path_vids = '/data/zming/datasets/SiW/SiW_release/Train/live'
path_depth = '/data/zming/datasets/SiW/SiW_release/Depth/Train/live'

subjects = os.listdir(path_vids)
subjects.sort()
for sub in subjects:
    if not os.path.isdir(os.path.join(path_depth, sub)):
        os.mkdir(os.path.join(path_depth, sub))
    videos = glob.glob(os.path.join(path_vids, sub, '*.mov'))
    videos.sort()
    for video in videos:
        vid_name = str.split(video, '/')[-1]
        depth_vid = os.path.join(path_depth, sub, vid_name[:-4])
        if not os.path.isdir(depth_vid):
            os.mkdir(depth_vid)

        cap = cv2.VideoCapture(video)
        frameCnt = 0
        while(True):
            ret, image = cap.read()
            if ret == False:
                break



            image_shape = [image.shape[0], image.shape[1]]

            pos = prn.process(image, None, None, image_shape)

            kpt = prn.get_landmarks(pos)

            # 3D vertices
            vertices = prn.get_vertices(pos)

            depth_scene_map = DepthImage.generate_depth_image(vertices, kpt, image.shape, isMedFilter=True)

            # cv2.imshow('IMAGE', image)
            # cv2.imshow('DEPTH', depth_scene_map)
            # cv2.waitKey(0)

            cv2.imwrite(os.path.join(depth_vid, '%04d.jpg'%frameCnt), depth_scene_map)

            frameCnt += 1