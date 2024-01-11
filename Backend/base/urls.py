"""
URL configuration for base project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from service.views import SignupAPIView, LoginView, VodrecView, VoddetailView, CtclView
from service.views import Vodrec_SimpleView, SearchVODView, ChartView, ChartsampleView, ChartsampleView
from service.views import login, status_check, login_suc
from .views import index

urlpatterns = [
    path("admin/", admin.site.urls),
    path('login/', login, name = 'login'),
    path('dummy/', index),
    path('scheck/', status_check, name = 'token_check'),
    # path("signup/", SignupAPIView.as_view(), name = 'signup'),
    path('login2/', LoginView.as_view(), name = 'login2'),
    path('login_suc/', login_suc, name = 'login_success'),
    path('vodrec/', VodrecView.as_view(), name = 'vodrec'),
    path('vod_detail/<int:vod_id>/', VoddetailView.as_view(), name = 'voddetail'),
    path('home/<str:ct_cl>/', CtclView.as_view(), name = 'ct_cl_home'),
    path('vodrec_simple/<int:btn_selected>', Vodrec_SimpleView.as_view(), name = 'vodrec_simple'),
    path('search/', SearchVODView.as_view(), name = 'vod_search'),
    path('chart/', ChartView.as_view(), name = 'see_chart'),
    path('chart_sample/', ChartsampleView.as_view(), name = 'chart_sample')
    # path('cb/', CBView.as_view(), name = 'cb')
]
