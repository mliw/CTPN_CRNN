from ctpn_tools.infer_core import CTPN  
from crnn_tools.infer_core import CRNN  
from glob import glob
import numpy as np
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Cut out the pictures
def cut_out(img,rect):
    return img[rect[1]:(rect[3]+1),rect[0]:(rect[2]+1)]


def clip_box(bbox, im_shape):
    # x1 >= 0
    bbox[0] = np.maximum(np.minimum(bbox[0], im_shape[1] - 1), 0)
    # y1 >= 0
    bbox[1] = np.maximum(np.minimum(bbox[1], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    bbox[2] = np.maximum(np.minimum(bbox[2], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    bbox[3] = np.maximum(np.minimum(bbox[3], im_shape[0] - 1), 0)

    return bbox
  

if __name__=="__main__":
    
    # 0 Define ctpn_model and crnn_model
    ctpn_model = CTPN()
    ctpn_model._load_weights("weights/CTPN.h5")
    crnn_model = CRNN()
    crnn_model._load_weights("weights/CRNN.h5")
    pictures = glob('asset/original_pictures/*')
    
    
    # 1 Load weights
    pic = 'asset/original_pictures\\pku_math.jpg'
    img = cv2.imread(pic)
    my_path = "asset"
    pic_name = pic.split("\\")[1]
    text_rects = ctpn_model.predict_origin(pic, my_path+"/"+pic_name)
    recognition_result = []
    pieces = []
    text_rects=sorted(text_rects,key=lambda x: ((x[1]+x[3])/2,(x[2]+x[0])/2))
    for i in range(len(text_rects)):
        piece = cut_out(img,clip_box(list(text_rects[i]),img.shape))
        recognition_result.append(crnn_model.predict_path(piece))
        pieces.append(piece)

