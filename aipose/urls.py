"""
URL configuration for aipose project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
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
from django.urls import include, path
from .views import SeatedPosture,HandPosition,DeskPosition,Annotation
from django.conf import settings
from django.conf.urls.static import static
from django.http import HttpResponse
from django.urls import re_path
from . import consumers

def home_view(request):
    return HttpResponse("Welcome to the homepage!")

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/images/seatedposture/', SeatedPosture.as_view(), name='image-list'),
    path('api/images/handposition/', HandPosition.as_view(), name='image-list'),
    path('api/images/deskposition/', DeskPosition.as_view(), name='image-list'),
    path('api/images/annotateimage/',Annotation.as_view(), name='image-list'),
    path('', home_view, name='home'),
    # re_path(r'ws/pose/$', consumers.PoseConsumer.as_asgi()),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
