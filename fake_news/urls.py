from django.contrib import admin
from django.urls import path
from . import views , lstmA , predict

urlpatterns = [
     path('', views.index, name='index'),
     path(r'predict', predict.predict),
]