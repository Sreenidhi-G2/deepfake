from django.contrib import admin
from django.urls import path, include
from detection import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('detection/', include('detection.urls')),
    path('', views.index, name='index'),
]
