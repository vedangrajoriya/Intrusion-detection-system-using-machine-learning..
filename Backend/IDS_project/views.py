import os
from datetime import datetime
from django.conf import settings

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny

from .serializers import URLAnalysisSerializer
from .models import URLAnalysis
from .ml_analyzer import XGBoostURLAnalyzer  # Updated import

class AnalyzeURLView(APIView):
    permission_classes = [AllowAny]
    
    # Initialize analyzer as a class variable to reuse across requests
    _analyzer = None
    
    @classmethod
    def get_analyzer(cls):
        """Get or create analyzer instance (singleton pattern)"""
        if cls._analyzer is None:
            cls._analyzer = XGBoostURLAnalyzer()
        return cls._analyzer
    
    def post(self, request):
        """Handle URL analysis requests"""
        serializer = URLAnalysisSerializer(data=request.data)
        
        if not serializer.is_valid():
            return Response({
                'error': 'Invalid URL provided',
                'details': serializer.errors
            }, status=status.HTTP_400_BAD_REQUEST)
        
        url = serializer.validated_data['url']
        
        try:
            # Get analyzer instance
            analyzer = self.get_analyzer()
            
            # Perform XGBoost ML analysis
            analysis_result = analyzer.analyze_url(url)
            
            # Check if analysis returned an error
            if 'error' in analysis_result:
                return Response({
                    'error': 'URL analysis failed',
                    'details': analysis_result.get('error', 'Unknown error')
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Save to database
            try:
                analysis = URLAnalysis.objects.create(
                    url=url,
                    security_status=analysis_result.get('security_status', 'Unknown'),
                    confidence=analysis_result.get('confidence', '0.00%'),
                    security_score=analysis_result.get('security_score', '0.00/100'),
                    risk_level=analysis_result.get('risk_level', 'Unknown')
                )
                analysis_time = analysis.analysis_time.strftime('%Y-%m-%d %H:%M:%S')
            except Exception as db_error:
                # If database save fails, continue with analysis but use current time
                print(f"Database save error: {db_error}")
                analysis_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Prepare response with all required fields
            response_data = {
                'security_status': analysis_result.get('security_status', 'Unknown'),
                'confidence': analysis_result.get('confidence', '0.00%'),
                'security_score': analysis_result.get('security_score', '0.00/100'),
                'risk_level': analysis_result.get('risk_level', 'Unknown'),
                'analysis_time': analysis_time,
                'ssl_security': analysis_result.get('ssl_security', {'status': 'Unknown'}),
                'header_security': analysis_result.get('header_security', {'status': 'Unknown'}),
                'content_security': analysis_result.get('content_security', {'status': 'Unknown'}),
                'recommendations': analysis_result.get('recommendations', ['Unable to generate recommendations'])
            }
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            # Log the error for debugging
            print(f"Analysis exception for URL {url}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Return a user-friendly error response
            return Response({
                'error': 'Analysis failed',
                'details': str(e),
                'url': url
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class HealthCheckView(APIView):
    """Health check endpoint to verify model is loaded"""
    permission_classes = [AllowAny]
    
    def get(self, request):
        """Check if the ML model is loaded and ready"""
        try:
            analyzer = AnalyzeURLView.get_analyzer()
            
            model_status = {
                'model_loaded': analyzer.model is not None,
                'scaler_loaded': analyzer.scaler is not None,
                'label_encoder_loaded': analyzer.label_encoder is not None,
                'features_count': len(analyzer.selected_features) if analyzer.selected_features else 0,
                'status': 'ready' if analyzer.model else 'fallback_mode'
            }
            
            return Response(model_status, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response({
                'status': 'error',
                'details': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class ModelInfoView(APIView):
    """Get information about the loaded model"""
    permission_classes = [AllowAny]
    
    def get(self, request):
        """Return model metadata"""
        try:
            analyzer = AnalyzeURLView.get_analyzer()
            
            info = {
                'model_type': 'XGBoost Classifier' if analyzer.model else 'Fallback Model',
                'selected_features': analyzer.selected_features,
                'feature_count': len(analyzer.selected_features) if analyzer.selected_features else 0,
                'model_loaded': analyzer.model is not None,
                'classes': analyzer.label_encoder.classes_.tolist() if analyzer.label_encoder else []
            }
            
            return Response(info, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response({
                'error': 'Failed to retrieve model info',
                'details': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)