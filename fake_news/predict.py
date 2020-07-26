
import sys
from django.http import HttpResponse , JsonResponse , HttpRequest
import numpy as np
import pandas as pd
from . import lstmA , lstmB
# from . import  bert
def predict(request):
    res_data = {}
    res_data['success'] = False 
    model_list = ['bert', 'lstmA' , 'lstmB']#pandas title
    if request.POST:
        ans = request.POST.get('input_text')
        # print(f'request:{ans}')
        res_data['input_text'] = ans
        #
        lstmA_data = lstmA.getLSTMPredict(ans)       
        lstmB_data = lstmB.getLSTMPredict(ans)
        #
        # bert_data = bert.getBERTPredict(ans)
        # if bert_data['success']:
        #     res_data['success'] = True 
        #     res_data['result'] = True
        #     res_data['confidence'] = 0.999

        # print(f'lstm_data:{lstm_data}')
        # vote start
        # box={
        #     "model":['lstmA' , 'lstmB'],
        #     "result":[lstmA_data['result'],lstmB_data['result']],
        #     "confidence":[lstmA_data['confidence'],lstmB_data['confidence']]
        # }
        model_list = ['lstmA' , 'lstmB']
        index_list = ['result','confidence']
        lstmA_data_list = list([lstmA_data['result'],lstmA_data['confidence']])
        lstmB_data_list = list([lstmB_data['result'],lstmB_data['confidence']]) 
        df = pd.DataFrame([lstmA_data_list,lstmB_data_list],index=model_list,columns=index_list)
        print(df)
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
