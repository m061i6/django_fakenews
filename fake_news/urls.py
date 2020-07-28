from django.contrib import admin
from django.urls import path
from . import views , lstmA , predict

urlpatterns = [
     path('', views.index, name='index'),
     #3模型版本
     path(r'predict', predict.predict),
     #2模型版本
     # path(r'predict', predict.predict2),
]