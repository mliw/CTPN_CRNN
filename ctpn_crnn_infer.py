from ctpn_tools.infer_core import CTPN  
from crnn_tools.infer_core import CRNN  
from glob import glob
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Cut out the pictures
def cut_out(img,rect):
    return img[rect[1]:(rect[3]+1),rect[0]:(rect[2]+1)]
    

if __name__=="__main__":
    
    # 0 Define ctpn_model and crnn_model
    ctpn_model = CTPN()
    ctpn_model._load_weights("weights/CTPN.h5")
    crnn_model = CRNN()
    crnn_model._load_weights("weights/CRNN.h5")
    pictures = glob('asset/original_pictures/*')
    
    
    # 1 Load weights
    pic = 'asset/original_pictures\\chinese_text.png'
    img = cv2.imread(pic)
    my_path = "asset"
    pic_name = pic.split("\\")[1]
    text_rects = test_model.predict(pic, my_path+"/"+pic_name)
    recognition_result = []
    pieces = []
    for i in range(len(text_rects)):
        piece = cut_out(img,text_rects[i])
        recognition_result.append(crnn_model.predict_path(piece))
        pieces.append(piece)

    