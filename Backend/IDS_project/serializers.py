from rest_framework import serializers
from .models import URLAnalysis

class URLAnalysisSerializer(serializers.Serializer):
    url = serializers.URLField()

class URLAnalysisResponseSerializer(serializers.ModelSerializer):
    ssl_security = serializers.SerializerMethodField()
    header_security = serializers.SerializerMethodField()
    content_security = serializers.SerializerMethodField()
    recommendations = serializers.SerializerMethodField()
    
    class Meta:
        model = URLAnalysis
        fields = [
            'security_status', 'confidence', 'security_score',
            'risk_level', 'analysis_time', 'ssl_security',
            'header_security', 'content_security', 'recommendations'
        ]
    
    def get_ssl_security(self, obj):
        return {"status": "Secure"}
    
    def get_header_security(self, obj):
        return {"status": "Needs Improvement"}
    
    def get_content_security(self, obj):
        return {"status": "Good"}
    
    def get_recommendations(self, obj):
        return [
            "Implement security headers (HSTS, CSP, X-Frame-Options)",
            "Enable content security policy",
            "Add X-Content-Type-Options header"
        ]