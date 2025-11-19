from django.db import models

class URLAnalysis(models.Model):
    url = models.URLField()
    security_status = models.CharField(max_length=20)
    confidence = models.CharField(max_length=10)
    security_score = models.CharField(max_length=20)
    risk_level = models.CharField(max_length=20)
    analysis_time = models.DateTimeField(auto_now_add=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Analysis: {self.url} - {self.risk_level}"