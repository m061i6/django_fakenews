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
    # print(request.POST)
    if request.POST:
        ans = request.POST.get('input_text')
        # print(f'request:{ans}')
        res_data['input_text'] = ans
        lstm_data = getLSTMPredict(ans)
        if lstm_data['success']:
            res_data['success'] = True 
            res_data['result'] = lstm_data['result']
            res_data['confidence'] = lstm_data['confidence']
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
        max_sequence_length = 500 # 最長序列長度為n個字
        min_word_frequency = 3 # 出現頻率小於n的話 ; 就當成罕見字
        
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
    
    # ans='又到了四年一度的閏年。在許多歐洲國家，2月29日是女性專屬的求婚日，在這天求婚不只不會被貼上標籤，男生如果拒絕還得受罰！女性選在閏年向男性求婚的習俗從何而來？最常見的說法是源自5世紀的愛爾蘭。據傳聖布麗姬（St. Brigid）向聖派翠克主教（St. Patrick）抱怨，女生等男生求婚要等很久，於是主教就規定在閏年的2月29日，女性可以求婚。後來這項規範傳入蘇格蘭，瑪格麗特皇后（Queen Margaret）據此頒布法令，讓蘇格蘭女性可以在229當天向男性求婚，而且對方如過拒絕還要受罰。男生如果拒絕求婚，就要給女方一個吻或一件絲綢洋裝，或是一雙手套。在部分上層歐洲社會中，還要給到12雙手套，這樣才能讓女生把手遮起來，避免被別人發現沒戴婚戒。至於為什麼會有罰男方獻吻的說法，則是因為有一說是聖派翠克主教一答應讓女生求婚，聖布麗姬立刻下跪向主教求婚，主教拒絕了，但給她一個吻並送她一件絲綢長袍。不過上述故事應該都是杜撰的。聖布麗姬不一定是真實存在的人物，就算真有其人，聖派翠克主教過世的時候，聖布麗姬也還是個未滿10歲的小女孩。瑪格麗特皇后則是7歲就過世了，不太可能真的立過這條法令。女生229才能求婚引發平權爭議不管實際上是怎麼開始的，這套傳統就此傳承了下來。但隨著兩性平權的意識越來越普及，女性選在閏年求婚的說法也遭受了抨擊。熟悉這個傳統的美國學者柏金（Katherine Parkin）就認為，在這個女性地位逐漸提升的年代，特別准許女生每四年可以求婚一次，實在可笑，甚至有羞辱之嫌。但也有人認為，這項傳統的存在其實也是在鼓勵女性拋開傳統枷鎖，當她們所愛的人不敢開口的時候，勇敢站出來主導情勢，從這點來看或許也沒有這麼違反女權的概念。實際上還真的蠻多女生挑在229求婚的。2008年，來自英國的梅特卡夫（Sally Metcalf）就選在那一天向長跑10年的男友求婚成功。她說，「哪一天都好，不過在2月29日訂婚確實讓我覺得我們蠻特別的。」她也鼓勵其他女性不要再等了，直接開口問，只要對方愛你，就不該拒絕。在愛爾蘭和芬蘭，也有很多人相信229是個求婚幸運日，愛爾蘭更有一說是在229求婚可以降低未來的離婚機率。可惜沒有統計數據證明在閏年結婚或訂婚的人比較有可能白頭偕老，在男女平等的時代，誰開口似乎也沒那麼重要。但是女孩，如果妳一直找不到機會或提不起勇氣，不妨就選在這個連假給親愛的他一個驚喜吧！閏年傳統番外篇：有人相信229是幸運日，但也有人認為229這個多餘的日子很邪門。依據希臘傳統，如果在閏年結婚，以後就會離婚。選在閏年離婚的人，這輩子再也沒辦法找到幸福了。參考資料：華爾街日報、Irish Central、HuffPost、timeanddate.com'
    ans = clean_text(str(ans))
    
    ans_seg=[]
    ans_seg.append((' '.join([j for j in jieba.cut_for_search(ans) if j not in stopwords])))
    # ans_seg.append((' '.join([j for j in jieba.cut(ans, cut_all=False) if j not in stopwords])))
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
                    self.y=tf.nn.softmax(y) 
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
        test_loss, test_acc, test_predict_label,y  = sess.run([model.cross_entropy, model.accuracy, model.y_pred_cls,model.y], feed_dict = feedData(ans, y,1.0 ,ans.shape[0], model))    
        confidence = float(np.max(y))
        print(ans)
        if test_predict_label==0:
            data['result'] = False
        else:
            data['result'] = True
        data['confidence'] = confidence
        print(f'判斷:{test_predict_label},信心:{confidence}')
        data['success'] = True 
        return data
