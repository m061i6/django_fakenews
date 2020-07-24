from django.http import HttpResponse , JsonResponse , HttpRequest
from django.shortcuts import render
from django.template import loader
import random

import jieba
import pandas as pd # 引用套件並縮寫為 pd
import os
import numpy as np
import re
import tensorflow as tf
import time
from datetime import timedelta
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

priject_dir = os.path.dirname(__file__)  # get current directory
module_dir = os.path.join(priject_dir, 'model')

def predict(request):
    res_data = {}
    res_data['success'] = False 
    print(request.POST)
    if request.POST:
        ans = request.POST.get('input_text')
        res_data['input_text'] = ans
        lstm_data = getLSTMPredict(ans)
        if lstm_data['success']:
            res_data['success'] = True 
            res_data['result'] = lstm_data['result']
            res_data['cinfidence'] = lstm_data['cinfidence']
        # bert_data = getBERTPredict(ans)
        # print(f'lstm_data:{lstm_data}')
    return JsonResponse(res_data)

def getLSTMPredict(ans):
    stopwords=[]
    data={}
    module_lstm_dir = os.path.join(module_dir, 'lstm')
    with open(os.path.join(module_lstm_dir, 'stop.txt'), 'r', encoding='utf-8') as f:
        for line in f:
            if len(line)>0:
                stopwords.append(line.strip())
    class Config():
        max_sequence_length = 800 # 最長序列長度為n個字
        min_word_frequency = 5 # 出現頻率小於n的話 ; 就當成罕見字
        
        vocab_size = None
        category_num = None
        
        choose_model = 'lstm' # 想要使用的模型 ex lstm; rnn; gru
        embedding_dim_size =300 # 詞向量維度
        num_layer = 1 # 層數
        num_units = [128] # 神經元
        learning_rate = 0.0001 # 學習率         
        keep_prob = 0.8 
        
        batch_size = 64 # mini-batch
        epoch_size = 30 # epoch
        
        save_path = os.path.join(module_lstm_dir, 'best_validation') # 模型儲存檔名
    
    config = Config()
    def clean_text(text_string):
        text_string = re.sub(r'[^\u4e00-\u9fa5]+', '', text_string)
        return(text_string)

    ans = clean_text(str(ans))

    ans_seg=[]
    ans_seg.append((' '.join([j for j in jieba.cut(ans, cut_all=False) if j not in stopwords])))
    len(ans_seg)

    y1=[1]
    y = tf.keras.utils.to_categorical(y1)

    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(config.max_sequence_length, min_frequency=config.min_word_frequency)
    ans = np.array(list(vocab_processor.fit_transform(ans_seg)))

    train_x, train_y = np.load(os.path.join(module_lstm_dir, 'train_x.npy')),  np.load(os.path.join(module_lstm_dir, 'train_y.npy'))
    config.vocab_size = sum(1 for line in open(os.path.join(module_lstm_dir, 'vocab.txt'),encoding='utf-8'))
    config.category_num = train_y.shape[1]
    class TextRNN(object):
        def __init__(self, config):
            self.config = config
            
            # 四個等待輸入的data
            self.batch_size = tf.placeholder(tf.int32, [] , name = 'batch_size')
            self.keep_prob = tf.placeholder(tf.float32, [], name = 'keep_prob')
            
            # Initial
            self.x = tf.placeholder(tf.int32, [None, self.config.max_sequence_length] , name = 'x')
            self.y_label = tf.placeholder(tf.float32, [None, self.config.category_num], name = 'y_label')
            self.choose_model = config.choose_model
            self.rnn()
        # Get LSTM Cell
        def cell(self, num_units):
            #BasicLSTMCell activity => default tanh
            if self.choose_model == "lstm":
                #可以設定peephole等屬性
                LSTM_cell = rnn.LSTMCell(num_units, initializer = tf.random_uniform_initializer(-0.1, 0.1,seed=2 )) 
            elif self.choose_model == "basic":
                #最基礎的，沒有peephole
                LSTM_cell = rnn.BasicLSTMCell(num_units = num_units, forget_bias = 1.0, state_is_tuple = True) 
            else:
                LSTM_cell = rnn.GRUCell(num_units)

            return rnn.DropoutWrapper(LSTM_cell, output_keep_prob = self.keep_prob)
        
        def rnn(self):
            """RNN模型"""
            # 詞向量映射
            with tf.device('/cpu:0'):
                embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim_size])
                embedding_inputs = tf.nn.embedding_lookup(embedding, self.x)
                
            # RNN Layers
            with tf.name_scope('layers'):
                with tf.name_scope('RNN'):
                    LSTM_cells = rnn.MultiRNNCell([self.cell(int(self.config.num_units[_])) for _ in range(self.config.num_layer)])
                    # x_shape = tf.reshape(self.x, [-1, self.config.truncate, self.config.vectorSize])
                    
                with tf.name_scope('output'):
                    init_state = LSTM_cells.zero_state(self.batch_size, dtype = tf.float32)
                    outputs, final_state = tf.nn.dynamic_rnn(LSTM_cells, inputs = embedding_inputs, 
                                                            initial_state = init_state, time_major = False, dtype = tf.float32)
                    
            # Output Layer
            with tf.name_scope('output_layer'):
                # 全連接層，後面接dropout以及relu激活
                fc1 = tf.layers.dense(outputs[:, -1, :], int(self.config.num_units[len(self.config.num_units)-1]))
                fc1 = tf.contrib.layers.dropout(fc1, self.keep_prob)
                fc1 = tf.nn.relu(fc1)
                    
                # 分類器
                y = tf.layers.dense(fc1, self.config.category_num, name = 'y')
            
            self.y_pred_cls = tf.argmax(y, axis = 1) #預測類別
            with tf.name_scope('cross_entropy'):
                with tf.name_scope('total'):
                    self.softmax = tf.nn.softmax_cross_entropy_with_logits(labels = self.y_label, logits = y)
                    self.cross_entropy = tf.reduce_mean(self.softmax)

            with tf.name_scope('train'):
                self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.cross_entropy)

            with tf.name_scope('accuracy'):
                with tf.name_scope('correct_prediction'):
                    self.correction_prediction = tf.equal(self.y_pred_cls, tf.argmax(self.y_label, axis = 1))
                with tf.name_scope('accuracy'):
                    self.accuracy = tf.reduce_mean(tf.cast(self.correction_prediction, tf.float32))
    def feedData(x_batch, y_batch, keep_prob, batch_size, model):
        feed_dict = {
            model.x: x_batch,
            model.y_label: y_batch,
            model.keep_prob: keep_prob,
            model.batch_size: batch_size
        }
        return feed_dict
    tf.reset_default_graph()
    model = TextRNN(config)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess = sess, save_path = config.save_path)  # 讀取保存的模型
        test_loss, test_acc, test_predict_label, softmax_result = sess.run([model.cross_entropy, model.accuracy, model.y_pred_cls, model.softmax], feed_dict = feedData(ans, y,1.0 ,ans.shape[0], model))    
        if test_predict_label==0:
            data['result'] = False
            data['cinfidence'] = str(float(softmax_result))
            # print('有'+str(float(softmax_result))+'是假新聞')
        else:
            data['result'] = True
            data['cinfidence'] = str(float(softmax_result))
            # print('有'+str(float(softmax_result))+'是真新聞唷')
        data['success'] = True 
        return data
