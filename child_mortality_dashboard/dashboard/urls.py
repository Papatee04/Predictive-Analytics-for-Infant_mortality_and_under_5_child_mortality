from django.urls import path
from . import views
from dashboard.views import *
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('success/', views.success, name='success'),
    path('assessments/', views.assessments_management, name='assessments_management'),
    path('assessments/<int:assessment_id>/update/', views.update_assessment, name='update_assessment'),
    path('assessments/<int:assessment_id>/delete/', views.delete_assessment, name='delete_assessment'),
    path('assessments/<int:assessment_id>/', views.assessment_details, name='assessment_details'),


    # Authentication URLs
    path('login/', login_view, name='account_login'),
    path('logout/', logout_view, name='account_logout'),
    path('signup/', signup_view, name='account_signup'),
    path('profile/', profile_edit_view, name='profile_edit'),
    
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
    path('assessments/<int:assessment_id>/details/', views.assessment_details, name='assessment_details'),     
]
