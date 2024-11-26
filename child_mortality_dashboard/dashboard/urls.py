from django.urls import path
from . import views
from dashboard.views import *
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('success/', views.success, name='success'),

    # Authentication URLs
    path('login/', login_view, name='account_login'),
    path('logout/', logout_view, name='account_logout'),
    path('signup/', signup_view, name='account_signup'),
    path('profile/', profile_view, name='profile_edit'),
    
    # Password change and reset URLs
    path('password_change/', 
         auth_views.PasswordChangeView.as_view(
             template_name='dashboard/password_change.html',
             success_url='/dashboard/password_change/done/'
         ), 
         name='account_change_password'),
    path('password_change/done/', 
         auth_views.PasswordChangeDoneView.as_view(
             template_name='dashboard/password_change_done.html'
         ), 
         name='password_change_done'),
]
