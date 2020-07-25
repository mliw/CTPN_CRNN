import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers 
from tensorflow.keras import backend as K
from crnn_tools import config
import time
import cv2
import warnings
warnings.filterwarnings("ignore")
CLASS_NUM = 5990
THRESHOLD = 4


with open("data/labels/char_std_5990.txt","r",encoding="utf-8") as f:
    total_list = f.readlines()
total_list = [items.replace("\n","") for items in total_list]
total_list = np.array(total_list)
TOTAL_LIST = total_list.copy()


def ctc_lambda_func(args):
    labels, y_pred, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def add_color(interval_img,mode=1):
    if mode==0: # Add left
        left_color = int(np.median(interval_img[:,0]))
        return_pic = left_color*np.ones((interval_img.shape[0],interval_img.shape[1]+8))
        return_pic[0:interval_img.shape[0],-interval_img.shape[1]:]=interval_img
        return_pic = return_pic.astype(interval_img.dtype)
        return return_pic

    elif mode==1: # Add right
        right_color = int(np.median(interval_img[:,-1]))
        return_pic = right_color*np.ones((interval_img.shape[0],interval_img.shape[1]+32))
        return_pic[0:interval_img.shape[0],0:interval_img.shape[1]]=interval_img
        return_pic = return_pic.astype(interval_img.dtype)
        return return_pic
    
    
def my_custom_loss(y_true, y_pred):
    return K.mean(y_pred)


def translate(pre):
    pre[pre==-1] = 0
    tem_list = TOTAL_LIST.copy()
    tem_list[0] = ""
    result = [''.join(tem_list[items]) for items in pre]
    return result

  
def cut(img_gray):
    blur = cv2.GaussianBlur(img_gray,(3,3),0)
    ret3,img_black = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    columns = np.sum(img_black,axis = 0)
    columns = (columns>0).astype(int)
    
    columns_extend = np.ones(len(columns)+2)
    columns_extend[1:-1]=columns
    columns_extend = np.array(columns_extend).astype(int)
    
    first_pos = columns_extend[1:]-columns_extend[:-1]
    first_point = np.where(first_pos==-1)[0]
    last_pos = columns_extend[:-1]-columns_extend[1:]
    last_point = np.where(last_pos==-1)[0]-1
    
    spaces = np.column_stack((first_point,last_point))
    intervals = spaces[(spaces[:,1]-spaces[:,0])>=THRESHOLD,:]
    results = []
    for interval in intervals:
        if interval[0]!=0 and interval[1]!=img_gray.shape[1]-1:
            results.append(interval)

    middles = [(result[0]+result[1])//2+1 for result in results]
    
    # for middle in middles:
    #     cv2.line(img_black,(middle,0),(middle,img_black.shape[0]-1),(255,0,0),1)
    # Image.fromarray(img_black)
    
    return middles


def resize_pic(path):
    img = cv2.imread(path)
    h,w,_ = img.shape
    if h!=32:
        ratio = 32/h
        w_new = w*ratio
        w_new = np.floor(w_new) if w_new-np.floor(w_new)<=np.floor(w_new)+1-w_new else np.floor(w_new)+1
        w_new = int(w_new)
        img = cv2.resize(img,(w_new,32),interpolation=cv2.INTER_CUBIC)
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    
    return img_gray

    
def load_picture(path):
    img_gray = resize_pic(path)   
    mid_0 = cut(img_gray)
    mid_1 = cut(255-img_gray)
    mid_combined = mid_0+mid_1
    mid_combined.append(0)
    mid_combined.append(img_gray.shape[1])
    mid_combined = sorted(list(set(mid_combined))) # The vertical lines to cut images
    
    # for middle in mid_combined:
    #     cv2.line(img_gray,(middle,0),(middle,img_gray.shape[0]-1),(255,0,0),1)
    # Image.fromarray(img_gray)
    
    result = []
    for i in range(len(mid_combined)-1):
        interval_img = img_gray[:,mid_combined[i]:mid_combined[i+1]]
        interval_img = add_color(interval_img)
        interval_img = np.expand_dims(interval_img,axis = 2)
        interval_img = np.array([interval_img],dtype=np.float32)
        result.append(interval_img)
    
    return result,img_gray
    
    
class CRNN:

    def __init__(self,):
        self._build_crnn()
        
        
    def _build_crnn(self,):
 
        # Input
        labels = tf.keras.Input(name='the_labels',shape=[config.LABEL_LENGTH], dtype='float32')
        input_length = tf.keras.Input(name='input_length', shape=[1], dtype='int64')
        label_length = tf.keras.Input(name='label_length', shape=[1], dtype='int64')
        vgg_input = tf.keras.Input(shape=(32,None,1),name='vgg_input', dtype='float32')
        
        # None stands for batch_size 
        # Vgg filters=64 input_shape (None,32,280,1) output_shape (None,16,140,64)
        l0 = layers.Conv2D(filters=64,kernel_size=(3,3),padding="same",name="conv1",activation="relu")(vgg_input)
        l0 = layers.MaxPooling2D(pool_size=(2, 2), name='max1')(l0)  

        # Vgg filters=128 input_shape (None,16,140,64) output_shape (None,8,70,128)
        l1 = layers.Conv2D(128, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal',activation="relu")(l0) 
        l1 = layers.MaxPooling2D(pool_size=(2, 2), name='max2')(l1)    
        
        # Vgg filters=256 input_shape (None,8,70,128) output_shape (None,4,70,256)
        l2 = layers.Conv2D(256, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal',activation="relu")(l1)  
        l2 = layers.Conv2D(256, (3, 3), padding='same', name='conv4', kernel_initializer='he_normal',activation="relu")(l2)  
        l2 = layers.MaxPooling2D(pool_size=(2,1), name='max3')(l2) 
        
        # Vgg filters=512 input_shape (None,4,70,256) output_shape (None,1,70,512)
        l3 = layers.Conv2D(512, (3, 3), padding='same', name='conv5', kernel_initializer='he_normal')(l2)
        l3 = layers.BatchNormalization()(l3)
        l3 = layers.Activation('relu')(l3)
        l3 = layers.Conv2D(512, (3, 3), padding='same', name='conv6', kernel_initializer='he_normal')(l3) 
        l3 = layers.BatchNormalization()(l3)
        l3 = layers.Activation('relu')(l3)
        l3 = layers.MaxPooling2D(pool_size=(2, 1), name='max4')(l3) 

        # Vgg filters=512 input_shape (None,2,70,512) output_shape (None,1,69,512)
        l4 = layers.Conv2D(512, (2, 2), padding='valid', kernel_initializer='he_normal', name='con7',activation="relu")(l3)  

        # Transform from CNN to RNN input_shape (None,1,69,512) output_shape (None,69,512)
        l5 = tf.squeeze(l4,axis = 1)

        # RNN layer input_shape (None,69,512) output_shape (None,69,512)
        l6 = layers.Bidirectional(layers.LSTM(256, return_sequences=True), name='blstm1')(l5)
        l6 = layers.Bidirectional(layers.LSTM(256, return_sequences=True), name='blstm2')(l6)     
        
        # Get prediction input_shape (None,69,512) output_shape (None,69,5991)
        l7 = layers.Dense(CLASS_NUM+1, name='blstm2_out', activation='softmax')(l6)
        self.predict_core = Model(vgg_input,l7)  

        # Get loss and train_core
        loss_out = layers.Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([labels, l7, input_length, label_length]) #(None, 1)
        self.train_core = Model([vgg_input, labels, input_length, label_length], loss_out)


    def _compile_net(self,opt):
        self.train_core.compile(loss={'ctc': lambda y_true, y_pred: y_pred},optimizer=opt)
        
    
    def predict_path(self,path):
        a1 = time.time()
        pieces = load_picture(path)[0]
        final =[]
        for piece in pieces:
            final.append(self.predict_and_translate(piece)[0])
        a2 = time.time()
        timing = np.round(a2-a1,3)
        print("="*90)
        print("file_path is:{}".format(path))
        print("result is:"+" ".join(final))
        print("elapsed time is:{}s".format(timing))
        print("="*90)
        print("")
        return " ".join(final)
    
        
    def predict_and_translate(self,test_data):
        pre = self.predict_core(tf.constant(test_data))
        seq_length = pre.shape[1]
        batch_size = pre.shape[0]
        sequence_length = np.ones(batch_size)*seq_length
        sequence_length = sequence_length.astype(np.int32)
        result = K.ctc_decode(tf.constant(pre), sequence_length, greedy=True)  
        return translate(result[0][0].numpy())   
    

    def _load_weights(self,path):
        self.train_core.load_weights(path)
        output = self.train_core.get_layer("blstm2_out").output
        img_input = self.train_core.input[0]
        self.predict_core = Model(img_input,output)  


