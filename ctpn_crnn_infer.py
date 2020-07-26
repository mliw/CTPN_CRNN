from ctpn_tools.infer_core import CTPN  
from crnn_tools.infer_core import CRNN  
from glob import glob
import numpy as np
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def cut_out(img,rect):
    return img[rect[1]:(rect[3]+1),rect[0]:(rect[2]+1)]


def clip_box(bbox, im_shape):
    bbox[0] = np.maximum(np.minimum(bbox[0], im_shape[1] - 1), 0)
    bbox[1] = np.maximum(np.minimum(bbox[1], im_shape[0] - 1), 0)
    bbox[2] = np.maximum(np.minimum(bbox[2], im_shape[1] - 1), 0)
    bbox[3] = np.maximum(np.minimum(bbox[3], im_shape[0] - 1), 0)
    return bbox

  
def renew(path):
    if not os.path.exists(path):
        os.mkdir(path)
    

def infer(crnn_model,ctpn_model,picture):
    img = cv2.imread(picture)
    output_path = picture.split("\\")[1].split(".")[0]
    renew("asset/"+output_path)
    cv2.imwrite("asset/"+output_path+"/original.jpg",img)
    text_rects = ctpn_model.predict(picture, "asset/"+output_path+"/detection.jpg")
    
    # Begin CRNN
    recognition_result = []
    for i in range(len(text_rects)):
        piece = cut_out(img,clip_box(list(text_rects[i]),img.shape))
        recognition_result.append(str(i+1)+": "+crnn_model.predict_path(piece)+"\n")
    final = "".join(recognition_result)
        
    # Output results
    with open("asset/"+output_path+"/result.txt","w") as f:
        f.writelines(final)
    
    
if __name__=="__main__":
    
    # 0 Define ctpn_model, crnn_model and load weights. Get pictures.
    ctpn_model = CTPN()
    ctpn_model._load_weights("weights/CTPN.h5")
    crnn_model = CRNN()
    crnn_model._load_weights("weights/CRNN.h5")
    pictures = glob('asset/original_pictures/*')


    # 1 Start inference  
    for picture in pictures:
        infer(crnn_model,ctpn_model,picture)

    
    



    