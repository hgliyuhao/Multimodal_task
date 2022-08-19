# -*- coding: utf-8 -*-
from __future__ import print_function

import keras
import numpy as np
import fairies as fa
from tqdm import tqdm
from keras.layers import *
from keras.models import *

from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.backend import keras, set_gelu, K
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.layers import LayerNormalization
import tensorflow as tf

from net import *
import cv2

import random 
random.seed(1) 
set_gelu('tanh')  # 切换gelu版本
maxlen = 32
batch_size = 32
p = 'chinese_roberta_wwm_ext_L-12_H-768_A-12/'
config_path = p +'bert_config.json'
checkpoint_path = p + 'bert_model.ckpt'
dict_path = p +'vocab.txt'
tokenizer = Tokenizer(dict_path, do_lower_case=True)
all_category = set()

def read_img(img_path):
    
    """
        涉及到后面图片模块的处理,因此img_size定为固定值
        单图读取函数（对非方形的图片进行白色填充，使其变为方形）
    """
    
    img_size = 224
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    if height > width:
            height, width = img_size, width * img_size // height
            img = cv2.resize(img, (width, height))
            delta = (height - width) // 2
            img = cv2.copyMakeBorder(
                img,
                top=0,
                bottom=0,
                left=delta,
                right=height - width - delta,
                borderType=cv2.BORDER_CONSTANT,
                value=[255, 255, 255]
            )
    else:
            height, width = height * img_size // width, img_size
            img = cv2.resize(img, (width, height))
            delta = (width - height) // 2
            img = cv2.copyMakeBorder(
                img,
                top=delta,
                bottom=width - height - delta,
                left=0,
                right=0,
                borderType=cv2.BORDER_CONSTANT,
                value=[255, 255, 255]
            )
    img = img.astype('float32')
    return img[..., ::-1] # cv2的读取模式为BGR，但keras的模型要求为RGB

def read_data(path):

    data = []
    all_data = fa.read_json(path)
    random.shuffle(all_data)

    for i in all_data:
        text = i['text']
        label = i['label']
        img_path = i['img_path']
        
        data.append([text,label,img_path])
        all_category.add(label)
    
    return data         


train_data = read_data('train_data/train.json')
val_data = read_data('train_data/val.json')

all_category = list(all_category)
all_category.sort()

id2label,label2id = fa.label2id(all_category)

class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        idxs = list(range(len(self.data)))
        if random:
            np.random.shuffle(idxs)
        batch_images, batch_token_ids, batch_segment_ids,batch_labels = [], [], [], []
        for i in idxs:
            data = self.data[i]
            text = data[0]
            label = label2id[data[1]]
            image_path = data[2]
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            batch_images.append(read_img(image_path))
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                batch_images = np.array(batch_images)
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids, batch_images], batch_labels
                batch_images, batch_token_ids, batch_segment_ids,batch_labels = [], [], [], []

def expand_dim(x):
    x1 = K.expand_dims(x,axis=-1)
    return x1
                
cv_model = build_cv_model(
    model_name = "resnet50",
    hidden_size = 768
)

cv_inputs = cv_model.input
cv_output = cv_model.output

bert = build_transformer_model(
    config_path,
    checkpoint_path,
    layer_norm_cond= cv_output,
    layer_norm_cond_hidden_size=64,
    layer_norm_cond_hidden_act='swish',
    additional_input_layers= cv_inputs,
)

output = Lambda(lambda x: x[:, 0],
                name='CLS-token')(bert.output)



fusion = LayerNormalization(conditional=True)([output, cv_output])
output = Dense(len(all_category),activation='softmax')(fusion)

model = Model(bert.inputs, output)
model.compile(

    loss='sparse_categorical_crossentropy',
    optimizer=Adam(2e-5),  # 用足够小的学习率
    metrics=['accuracy'],

)
model.summary()

def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        res = model.predict(x_true)
        y_pred = res.argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)

        right += (y_true == y_pred).sum()
    return right / total


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0:
            self.best_val_acc = 0.
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
#             model.save_weights('model/muti_model.hdf5')
        print(u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n'
              % (val_acc, self.best_val_acc, 0))


if __name__ == '__main__':

    evaluator = Evaluator()
    train_generator = data_generator(train_data, batch_size)
    valid_generator = data_generator(val_data, batch_size)

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=60,
        callbacks=[evaluator]
    )

