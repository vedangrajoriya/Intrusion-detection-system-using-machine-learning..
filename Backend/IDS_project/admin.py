from django.contrib import admin
from .models import URLAnalysis

@admin.register(URLAnalysis)
class URLAnalysisAdmin(admin.ModelAdmin):
    list_display = ['url', 'risk_level', 'security_score', 'analysis_time']
    list_filter = ['risk_level', 'analysis_time']
    search_fields = ['url']
    readonly_fields = ['analysis_time', 'created_at']