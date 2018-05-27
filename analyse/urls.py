from django.urls import path

from . import views

urlpatterns = [
    path('index/', views.index_view, name='index'),
    path('products/', views.products_view, name='products'),
    path('dataset/', views.dataset_view, name='dataset'),
    path('contact/', views.contact_view, name='contact'),
    path('singlepoint/', views.spd_view, name='spd'),
    path('register/',views.register,name = 'register')
    # path('thisurl/', views.process_spd_data, name='process'),  # 处理数据的url, 当前页面的地址
    # path('progressurl/', views.show_spd_progress, name='progress'),  # 查询进度的url, 不需要html页面
]