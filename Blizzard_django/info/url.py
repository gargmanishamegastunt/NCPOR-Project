from . import views
from django.urls import path
urlpatterns = [

    path('', views.home, name='home'),
    path('year/', views.year, name='year'),
    path('month/', views.month, name='month'),
    path('prediction/', views.pred, name ='prediction'),
    # path('cf/', views.cf, name='cf')

]