from django.http import HttpResponse , JsonResponse , HttpRequest
from . import lstmA
def predict(request):
    res_data = {}
    res_data['success'] = False 
    if request.POST:
        ans = request.POST.get('input_text')
        # print(f'request:{ans}')
        res_data['input_text'] = ans
        lstm_data = lstmA.getLSTMPredict(ans)
        if lstm_data['success']:
            res_data['success'] = True 
            res_data['result'] = lstm_data['result']
            res_data['confidence'] = lstm_data['confidence']
        # bert_data = getBERTPredict(ans)
        # print(f'lstm_data:{lstm_data}')
    return JsonResponse(res_data)
