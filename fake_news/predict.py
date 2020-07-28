
import sys
from django.http import HttpResponse , JsonResponse , HttpRequest
import numpy as np
import pandas as pd
from . import lstmA , lstmB , bert

def predict(request):
    res_data = {}
    res_data['success'] = False 
    if request.POST:
        ans = request.POST.get('input_text')
        # print(f'request:{ans}')
        res_data['input_text'] = ans
        #
        lstmA_data = lstmA.getLSTMPredict(ans)
        print('====lstmA_data..done=====')       
        lstmB_data = lstmB.getLSTMPredict(ans)
        print('=====lstmB_data..done=====')  
        bert_data = bert.getBERTPredict(ans)
        print('=====bert_data..done=====')

        model_list = ['bert', 'lstmA' , 'lstmB']
        index_list = ['result','confidence']
        lstmA_data_list = list([lstmA_data['result'],lstmA_data['confidence']])
        lstmB_data_list = list([lstmB_data['result'],lstmB_data['confidence']]) 
        bert_data_list = list([bert_data['result'],bert_data['confidence']]) 
        df = pd.DataFrame([bert_data_list,lstmA_data_list,lstmB_data_list],index=model_list,columns=index_list)
        print('==========================')
        print(df)
        print('==========================')
        if df['result'].sum() > 1:#True 兩票以上
            result = df[df['result']==True]
            vote_result = True
        else:
            result = df[df['result']==False]
            vote_result = False
        vote_confidence = result.iloc[result['confidence'].argmax()]['confidence']
        print(f'投票結果為:{vote_result},信心指數為:{vote_confidence}')
        try:
            res_data['result'] = vote_result
            res_data['confidence'] = vote_confidence
        except:
            res_data['error'] = "資料判斷發生錯誤，請稍後再試"
        res_data['success'] = True 
        print(f'res_data:{res_data}')
    else:
        res_data['error'] = "資料傳輸發生錯誤，請稍後再試"
        res_data['success'] = True 
    return JsonResponse(res_data)
def predict2(request):
    res_data = {}
    res_data['success'] = False 
    model_list = ['bert', 'lstmA' , 'lstmB']#pandas title
    if request.POST:
        ans = request.POST.get('input_text')
        res_data['input_text'] = ans
        lstmA_data = lstmA.getLSTMPredict(ans)    
        print('====lstmA_data..done=====')     
        lstmB_data = lstmB.getLSTMPredict(ans)
        print('=====lstmB_data..done=====')

        model_list = ['lstmA' , 'lstmB']
        index_list = ['result','confidence']
        lstmA_data_list = list([lstmA_data['result'],lstmA_data['confidence']])
        lstmB_data_list = list([lstmB_data['result'],lstmB_data['confidence']]) 
        df = pd.DataFrame([lstmA_data_list,lstmB_data_list],index=model_list,columns=index_list)
        if df.loc['lstmA'].confidence > 0.86:
            result = df.loc['lstmA']
        else:
            result = df.iloc[df['confidence'].argmax()]
        # result = df.loc[df['confidence'].argmax()]
        print(result)
        try:
            res_data['result'] = int(result.result.item())
            res_data['confidence'] = result.confidence.item()
        except:
            res_data['error'] = "資料判斷發生錯誤，請稍後再試"
        res_data['success'] = True 
        print(f'res_data:{res_data}')
    else:
        res_data['error'] = "資料傳輸發生錯誤，請稍後再試"
        res_data['success'] = True 
    return JsonResponse(res_data)
