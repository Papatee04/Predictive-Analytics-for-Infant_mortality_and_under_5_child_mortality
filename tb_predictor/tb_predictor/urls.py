# tb_predictor/urls.py (root urls.py)
from django.contrib import admin
from django.urls import path, include
from django.views.generic import TemplateView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('api.urls')),  # Include api URLs
    path('', TemplateView.as_view(template_name='api/dashboard.html'),
         name='dashboard'),  # Dashboard URL
]
