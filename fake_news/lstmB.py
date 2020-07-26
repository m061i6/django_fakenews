import os
import re

import pandas as pd
import numpy as np
import tensorflow as tf
import jieba
import opencc

priject_dir = os.path.dirname(__file__)  # get current directory
module_dir = os.path.join(priject_dir, 'model')

def getLSTMPredict(ans):
    data={}
    module_lstm_dir = os.path.join(module_dir, 'lstmB')
    jieba.set_dictionary(os.path.join(module_lstm_dir, 'dict_v2.txt'))
    
    with open(os.path.join(module_lstm_dir, 'stopwords_only_symbol_v2.txt'), 'r', encoding='utf8') as f:
        stops_symbol = f.read().split('\n')
    input_str = ans # 輸入新聞標題
    # print(f'input_str:{input_str}')
    converter = opencc.OpenCC('s2twp.json')
    s2twp_str = converter.convert(input_str)
    # print(f's2twp_str:{s2twp_str}')
    jieba_str = ' '.join([t for t in jieba.cut_for_search(str(s2twp_str)) if t not in stops_symbol])
    input_data_np = np.array([jieba_str])
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(os.path.join(module_lstm_dir, 'search_jieba_no_stopwords_train_vocab.pickle'))
    input_data_pd = np.array(list(vocab_processor.transform(input_data_np)))
    tf.reset_default_graph()
    saver = tf.train.import_meta_graph(os.path.join(module_lstm_dir, 'search_jieba_no_stopwords_train_vocab.ckpt.meta'))
    with tf.Session() as sess:
        saver.restore(sess, os.path.join(module_lstm_dir, 'search_jieba_no_stopwords_train_vocab.ckpt'))
        prob_and_ans = {"Placeholder:0": input_data_pd, "Placeholder_2:0": 1}
        prob = sess.run("probability:0", feed_dict = prob_and_ans)
        ans = sess.run("ans:0", feed_dict = prob_and_ans)
        # print(f'probability: {prob}') # 印出較高的機率
        # print(f'ans: {ans}') # 印出真或假( 1為真, 0為假)
        if ans[0].item()==0:
            data['result'] = False
        else:
            data['result'] = True
        data['confidence'] = prob[0].item()
        # print(f'判斷:{ans},信心:{prob}')
        # print(f'ans:{type(ans[0])},prob:{type(prob[0])}')
        data['success'] = True 
        return data
        

