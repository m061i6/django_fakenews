from django.http import HttpResponse , JsonResponse
from django.shortcuts import render
from django.template import loader
# def index(request):
#      return HttpResponse("Hello, world")
def index(request):
    return render(request,'index.html')