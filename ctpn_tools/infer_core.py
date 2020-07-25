import cv2
import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers 
from tensorflow.keras.optimizers import SGD
from ctpn_tools import libs
import time
from ctpn_tools.text_connector.text_proposal_connector_oriented import TextProposalConnectorOriented
VGG_WEIGHTS_PATH = "weights/imagenet_vgg16.h5"
IMAGE_MEAN = [123.68, 116.779, 103.939]
IOU_SELECT = 0.7


def draw_rect(rect, img):
    cv2.line(img, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 0), 2)
    cv2.line(img, (rect[2], rect[3]), (rect[6], rect[7]), (255, 0, 0), 2)
    cv2.line(img, (rect[6], rect[7]), (rect[4], rect[5]), (255, 0, 0), 2)
    cv2.line(img, (rect[4], rect[5]), (rect[0], rect[1]), (255, 0, 0), 2)
    

# 1 Define loss functions
def _rpn_loss_regr(y_true, y_pred):
    """
    smooth L1 loss
    
    y_true [1][HXWX10][3] (class,regr)
    y_pred [1][HXWX10][2] (reger)
    H and W are dimensions of feature map!
    """
    sigma = 9.0
    cls = y_true[0, :, 0]
    regr = y_true[0, :, 1:3]
    regr_keep = tf.where(K.equal(cls, 1))[:, 0]
    regr_true = tf.gather(regr, regr_keep)
    regr_true = tf.cast(regr_true,dtype=tf.float32) 
    regr_pred = tf.gather(y_pred[0], regr_keep)
    diff = tf.abs(regr_true - regr_pred)
    less_one = tf.cast(tf.less(diff, 1.0 / sigma), 'float32')
    loss = less_one * 0.5 * diff ** 2 * sigma + tf.abs(1 - less_one) * (diff - 0.5 / sigma)
    loss = K.sum(loss, axis=1)

    return K.switch(tf.size(loss) > 0, K.mean(loss), K.constant(0.0))


def _rpn_loss_cls(y_true, y_pred):
    """
    softmax loss

    y_true [1][1][HXWX10] class
    y_pred [1][HXWX10][2] class
    H and W are dimensions of feature map!
    """
    y_true = y_true[0][0]
    cls_keep = tf.where(tf.not_equal(y_true, -1))[:, 0]
    cls_true = tf.gather(y_true, cls_keep)
    cls_pred = tf.gather(y_pred[0], cls_keep)
    cls_true = tf.cast(cls_true, 'int64')
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=cls_true, logits=cls_pred)
    return K.switch(tf.size(loss) > 0, K.clip(K.mean(loss), 0, 10), K.constant(0.0))


# 2 Define model
class CTPN:

    def __init__(self,):
        self._build_net()

    def _build_net(self):
        vgg_model = VGG16(weights=None, include_top=False, input_shape=(None,None,3))
        vgg_model.load_weights(VGG_WEIGHTS_PATH)    
                
        #Start building model
        original_input = vgg_model.input
        sub_output = vgg_model.get_layer('block5_conv3').output
        x = layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu',
                   name='rpn_conv1')(sub_output)
        x = tf.squeeze(x,axis = 0)
        time_x = layers.Bidirectional(layers.GRU(128, return_sequences=True), name='blstm')(x)
        time_x = tf.expand_dims(time_x,axis=0)
        fc = layers.Conv2D(512, (1, 1), padding='same', activation='relu', name='lstm_fc')(time_x)     
        class_logit = layers.Conv2D(10 * 2, (1, 1), padding='same', activation='linear', name='rpn_class_origin')(fc)
        regression = layers.Conv2D(10 * 2, (1, 1), padding='same', activation='linear', name='rpn_regress_origin')(fc)
        class_logit_reshape = layers.Reshape(target_shape=(-1, 2), name='rpn_class_reshape')(class_logit)
        regression_reshape = layers.Reshape(target_shape=(-1, 2), name='rpn_regress_reshape')(regression)
        train_model = Model(original_input,[class_logit_reshape, regression_reshape])
        self.core = train_model

    def _compile_net(self,lr):
        self.core.compile(optimizer=SGD(lr),
                       loss={'rpn_regress_reshape': _rpn_loss_regr, 'rpn_class_reshape': _rpn_loss_cls},
                       loss_weights={'rpn_regress_reshape': 1.0, 'rpn_class_reshape': 1.0})
        
        
    def _load_weights(self,path):
        self.core.load_weights(path)
        
        
    def predict(self, image, output_path):
        print("="*60)
        st = time.time()
        if type(image) == str:
            img = cv2.imdecode(np.fromfile(image, dtype=np.uint8), cv2.IMREAD_COLOR)
        else:
            img = image
        h, w, c = img.shape
    
        # image size length must be greater than or equals 16 x 16, because of the image will be reduced by 16 times.
        if h < 16 or w < 16:
            transform_w = max(16, w)
            transform_h = max(16, h)
            transform_img = np.ones(shape=(transform_h, transform_w, 3), dtype='uint8') * 255
            transform_img[:h, :w, :] = img
            h = transform_h
            w = transform_w
            img = transform_img
    
        # zero-center by mean pixel
        m_img = img - IMAGE_MEAN
        m_img = np.expand_dims(m_img, axis=0)
        cls, regr = self.core.predict_on_batch(m_img)
        cls_prod = layers.Softmax()(cls)
        anchor = libs.gen_anchor((int(h / 16), int(w / 16)), 16)
        bbox = libs.bbox_transfor_inv(anchor, regr)
        bbox = libs.clip_box(bbox, [h, w])
    
        # score > 0.7
        fg = np.where(cls_prod[0, :, 1] > libs.IOU_SELECT)[0]
        select_anchor = bbox[fg, :]
        select_score = cls_prod.numpy()[0, fg, 1]
        select_anchor = select_anchor.astype('int32')
    
        # filter size
        keep_index = libs.filter_bbox(select_anchor, 16)
            
        # nsm
        select_anchor = select_anchor[keep_index]
        select_score = select_score[keep_index]
        select_score = np.reshape(select_score, (select_score.shape[0], 1))
        nmsbox = np.hstack((select_anchor, select_score))
        keep = libs.nms(nmsbox, 1 - libs.IOU_SELECT)
        select_anchor = select_anchor[keep]
        select_score = select_score[keep]
        
        # text line
        textConn = TextProposalConnectorOriented()
        text = textConn.simple_get_text_lines(select_anchor, select_score, [h, w])
        text = text.astype('int32')
        ed = time.time()
        print("Infer time is {}s".format(ed-st))
        print("="*60)
        # visualize
        for i in text:
            cv2.rectangle(img,(i[0],i[1]),(i[2],i[3]),color=(255,0,0))
        cv2.imwrite(output_path, img)

