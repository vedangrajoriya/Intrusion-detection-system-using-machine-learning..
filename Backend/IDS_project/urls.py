# ids_api/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('analyze-url/', views.AnalyzeURLView.as_view(), name='analyze-url'),
    path('health/', views.HealthCheckView.as_view(), name='health-check'),
    path('model-info/', views.ModelInfoView.as_view(), name='model-info'),
]