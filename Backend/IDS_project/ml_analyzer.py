# ids_api/ml_analyzer.py

import joblib
import pandas as pd
import numpy as np
import requests
import ssl
import socket
import re
from urllib.parse import urlparse
from datetime import datetime
import os
from django.conf import settings
import validators
from bs4 import BeautifulSoup
import warnings
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

warnings.filterwarnings('ignore')

class URLAnalyzerConfig:
    TIMEOUT = 10
    MAX_RETRIES = 2
    MAX_CONTENT_LENGTH = 2 * 1024 * 1024  # 2MB limit for Django
    SSL_VERIFY = False  # Set to True in production

class XGBoostURLAnalyzer:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.selected_features = []
        self.config = URLAnalyzerConfig()
        self.load_model()
    
    def load_model(self):
        """Load the trained XGBoost model and preprocessing objects"""
        try:
            model_dir = os.path.join(settings.BASE_DIR, 'ml_model')
            
            # Load model components
            model_path = os.path.join(model_dir, 'xgboost_model.pkl')
            scaler_path = os.path.join(model_dir, 'scaler.pkl')
            le_path = os.path.join(model_dir, 'label_encoder.pkl')
            features_path = os.path.join(model_dir, 'selected_features.pkl')
            
            if all(os.path.exists(p) for p in [model_path, scaler_path, le_path, features_path]):
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                self.label_encoder = joblib.load(le_path)
                self.selected_features = joblib.load(features_path)
                print("XGBoost model loaded successfully")
                print(f"Model type: {type(self.model)}")
                print(f"Selected features: {len(self.selected_features)}")
                print(f"Classes: {self.label_encoder.classes_}")
            else:
                print("Model files not found, using fallback prediction")
                print(f"Searched in: {model_dir}")
                print(f"Model exists: {os.path.exists(model_path)}")
                print(f"Scaler exists: {os.path.exists(scaler_path)}")
                print(f"Label encoder exists: {os.path.exists(le_path)}")
                print(f"Features exists: {os.path.exists(features_path)}")
                
                self.model = None
                # Define fallback features based on your ML model
                self.selected_features = [
                    'url_length', 'domain_length', 'has_https', 'num_dots',
                    'suspicious_tld', 'has_suspicious_words', 'subdomain_count',
                    'is_ip', 'has_hsts', 'has_csp'
                ]
                
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
    
    def create_session(self):
        """Create a session with retry strategy"""
        session = requests.Session()
        retry_strategy = Retry(
            total=self.config.MAX_RETRIES,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def extract_url_features(self, url):
        """Extract comprehensive features from URL"""
        try:
            if not validators.url(url):
                raise ValueError("Invalid URL format")
            
            features = {}
            parsed = urlparse(url)
            domain = parsed.netloc or parsed.path.split('/')[0]
            
            # Basic URL features
            features.update({
                'url_length': len(url),
                'domain_length': len(domain),
                'path_length': len(parsed.path),
                'num_dots': url.count('.'),
                'num_digits': sum(c.isdigit() for c in url),
                'num_special_chars': len(re.findall(r'[^a-zA-Z0-9.]', url)),
                'has_https': int(parsed.scheme == 'https'),
                'subdomain_count': len(domain.split('.')) - 2 if domain.count('.') > 1 else 0,
                'path_depth': len([x for x in parsed.path.split('/') if x]),
                'query_length': len(parsed.query),
                'fragment_length': len(parsed.fragment),
                'has_port': int(bool(parsed.port)),
                'is_ip': int(bool(re.match(r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$', domain))),
                'suspicious_tld': int(domain.split('.')[-1] in ['xyz', 'top', 'pw', 'cc', 'tk'] if '.' in domain else 0),
                'has_suspicious_words': int(bool(re.search(
                    r'login|admin|bank|secure|account|verify|confirm|update|payment|wallet|crypto',
                    url.lower()
                ))),
                'char_entropy': self.calculate_entropy(url)
            })
            
            # Try to get additional features from HTTP request
            session = self.create_session()
            try:
                response = session.get(
                    url, 
                    timeout=self.config.TIMEOUT,
                    verify=self.config.SSL_VERIFY,
                    headers={'User-Agent': 'Mozilla/5.0 (compatible; SecurityAnalyzer/1.0)'},
                    stream=True
                )
                
                # Limit content size
                content = b''
                for chunk in response.iter_content(chunk_size=8192):
                    content += chunk
                    if len(content) > self.config.MAX_CONTENT_LENGTH:
                        break
                
                # Parse content
                soup = BeautifulSoup(content, 'html.parser')
                
                # Content analysis features
                features.update({
                    'has_password_field': int(bool(soup.find('input', {'type': 'password'}))),
                    'has_form': int(bool(soup.find('form'))),
                    'external_links': len([a for a in soup.find_all('a', href=True) 
                                         if a['href'].startswith('http')]),
                    'iframe_count': len(soup.find_all('iframe')),
                    'script_count': len(soup.find_all('script')),
                    'hidden_elements': len(soup.find_all(style=re.compile(r'display:\s*none'))),
                    'redirect_count': len(response.history),
                    'response_time': response.elapsed.total_seconds(),
                    'content_length': len(content),
                    'status_code': response.status_code
                })
                
                # Security headers
                headers = response.headers
                features.update({
                    'has_hsts': int('strict-transport-security' in headers),
                    'has_xframe': int('x-frame-options' in headers),
                    'has_csp': int('content-security-policy' in headers),
                    'has_xss_protection': int('x-xss-protection' in headers),
                    'server_disclosed': int('server' in headers)
                })
                
            except Exception as e:
                print(f"HTTP request failed: {e}")
                # Set default values if request fails
                default_features = {
                    'has_password_field': 0, 'has_form': 0, 'external_links': 0,
                    'iframe_count': 0, 'script_count': 0, 'hidden_elements': 0,
                    'redirect_count': 0, 'response_time': 0, 'content_length': 0,
                    'status_code': 0, 'has_hsts': 0, 'has_xframe': 0, 'has_csp': 0,
                    'has_xss_protection': 0, 'server_disclosed': 0
                }
                features.update(default_features)
            
            # SSL Certificate analysis
            if parsed.scheme == 'https':
                try:
                    ctx = ssl.create_default_context()
                    ctx.check_hostname = False
                    ctx.verify_mode = ssl.CERT_NONE
                    
                    with ctx.wrap_socket(socket.socket(), server_hostname=domain) as s:
                        s.settimeout(5)
                        s.connect((domain, 443))
                        cert = s.getpeercert()
                        
                        if cert:
                            not_after = datetime.strptime(cert['notAfter'], r'%b %d %H:%M:%S %Y %Z')
                            days_valid = (not_after - datetime.now()).days
                            
                            features.update({
                                'cert_valid_days': max(0, days_valid),
                                'cert_is_valid': int(days_valid > 0),
                                'cert_issuer_trusted': int(any(
                                    trusted in str(cert.get('issuer', ''))
                                    for trusted in ['Let\'s Encrypt', 'DigiCert', 'GeoTrust', 'Comodo']
                                ))
                            })
                        else:
                            features.update({'cert_valid_days': 0, 'cert_is_valid': 0, 'cert_issuer_trusted': 0})
                            
                except Exception as e:
                    print(f"SSL check failed: {e}")
                    features.update({'cert_valid_days': 0, 'cert_is_valid': 0, 'cert_issuer_trusted': 0})
            else:
                features.update({'cert_valid_days': 0, 'cert_is_valid': 0, 'cert_issuer_trusted': 0})
            
            return pd.DataFrame([features])
            
        except Exception as e:
            print(f"Error extracting features for {url}: {e}")
            return None
    
    def calculate_entropy(self, s):
        """Calculate Shannon entropy of string"""
        if not s:
            return 0
        char_counts = {}
        for c in s:
            char_counts[c] = char_counts.get(c, 0) + 1
        
        entropy = 0
        length = len(s)
        for count in char_counts.values():
            p = count / length
            entropy -= p * np.log2(p)
        
        return entropy
    
    def calculate_security_score(self, features_df):
        """Calculate security score based on features"""
        weights = {
            'has_https': 15,
            'has_hsts': 10,
            'has_csp': 10,
            'has_xframe': 8,
            'has_xss_protection': 5,
            'cert_is_valid': 12,
            'cert_issuer_trusted': 8,
            'suspicious_tld': -15,
            'has_suspicious_words': -10,
            'is_ip': -8,
            'server_disclosed': -3
        }
        
        score = 50  # Base score
        for feature, weight in weights.items():
            if feature in features_df.columns:
                score += weight * features_df[feature].iloc[0]
        
        return max(0, min(100, score))
    
    def get_risk_level(self, security_score):
        """Determine risk level based on security score"""
        if security_score >= 80:
            return "Low Risk"
        elif security_score >= 60:
            return "Medium Risk"
        elif security_score >= 40:
            return "High Risk"
        else:
            return "Critical Risk"
    
    def analyze_url(self, url):
        """Main analysis function"""
        try:
            # Extract features
            features_df = self.extract_url_features(url)
            if features_df is None:
                return self.get_error_response(url, "Feature extraction failed")
            
            # Calculate security score
            security_score = self.calculate_security_score(features_df)
            
            # Prepare features for ML prediction
            prediction_features = pd.DataFrame(0, index=[0], columns=self.selected_features)
            for feature in self.selected_features:
                if feature in features_df.columns:
                    prediction_features[feature] = features_df[feature]
                else:
                    print(f"Warning: Feature '{feature}' not found, using default value 0")
            
            # Make prediction
            if self.model is not None and self.scaler is not None:
                try:
                    X_scaled = self.scaler.transform(prediction_features.values)
                    prediction_prob = self.model.predict_proba(X_scaled)
                    prediction = self.model.predict(X_scaled)[0]
                    confidence = np.max(prediction_prob) * 100
                    
                    if self.label_encoder is not None:
                        prediction_label = self.label_encoder.inverse_transform([prediction])[0]
                    else:
                        prediction_label = "Suspicious" if prediction == 1 else "Legitimate"
                        
                except Exception as e:
                    print(f"Prediction error: {e}")
                    prediction_label = "Suspicious" if security_score < 60 else "Legitimate"
                    confidence = 75.0
            else:
                # Fallback prediction based on security score
                prediction_label = "Legitimate" if security_score >= 60 else "Suspicious"
                confidence = min(security_score + 20, 95)
            
            # Security analysis
            ssl_status = self.analyze_ssl_security(features_df)
            header_status = self.analyze_header_security(features_df)
            content_status = self.analyze_content_security(features_df)
            
            # Generate recommendations
            recommendations = self.generate_recommendations(ssl_status, header_status, content_status, features_df)
            
            return {
                'security_status': str(prediction_label),
                'confidence': f"{confidence:.2f}%",
                'security_score': f"{security_score:.2f}/100",
                'risk_level': self.get_risk_level(security_score),
                'ssl_security': ssl_status,
                'header_security': header_status,
                'content_security': content_status,
                'recommendations': recommendations
            }
            
        except Exception as e:
            print(f"Analysis error for {url}: {e}")
            import traceback
            traceback.print_exc()
            return self.get_error_response(url, str(e))
    
    def analyze_ssl_security(self, features_df):
        """Analyze SSL security"""
        if features_df['has_https'].iloc[0] and features_df.get('cert_is_valid', pd.Series([0])).iloc[0]:
            return {'status': 'Secure'}
        elif features_df['has_https'].iloc[0]:
            return {'status': 'Needs Improvement'}
        else:
            return {'status': 'Insecure'}
    
    def analyze_header_security(self, features_df):
        """Analyze security headers"""
        security_headers = [
            features_df.get('has_hsts', pd.Series([0])).iloc[0],
            features_df.get('has_csp', pd.Series([0])).iloc[0],
            features_df.get('has_xframe', pd.Series([0])).iloc[0],
            features_df.get('has_xss_protection', pd.Series([0])).iloc[0]
        ]
        
        score = sum(security_headers)
        if score >= 3:
            return {'status': 'Good'}
        elif score >= 1:
            return {'status': 'Needs Improvement'}
        else:
            return {'status': 'Poor'}
    
    def analyze_content_security(self, features_df):
        """Analyze content security"""
        risk_factors = [
            features_df['has_suspicious_words'].iloc[0],
            features_df['suspicious_tld'].iloc[0],
            features_df['is_ip'].iloc[0]
        ]
        
        risk_score = sum(risk_factors)
        if risk_score == 0:
            return {'status': 'Good'}
        elif risk_score <= 1:
            return {'status': 'Moderate'}
        else:
            return {'status': 'Poor'}
    
    def generate_recommendations(self, ssl_status, header_status, content_status, features_df):
        """Generate security recommendations"""
        recommendations = []
        
        if ssl_status['status'] != 'Secure':
            recommendations.append("Implement HTTPS with a valid SSL certificate")
        
        if header_status['status'] in ['Poor', 'Needs Improvement']:
            recommendations.append("Implement security headers (HSTS, CSP, X-Frame-Options)")
            recommendations.append("Add Content Security Policy to prevent XSS attacks")
        
        if content_status['status'] in ['Poor', 'Moderate']:
            recommendations.append("Review URL structure and remove suspicious elements")
            
        if features_df['is_ip'].iloc[0]:
            recommendations.append("Use a proper domain name instead of IP address")
            
        if features_df['suspicious_tld'].iloc[0]:
            recommendations.append("Consider using a reputable domain TLD")
        
        if not recommendations:
            recommendations.append("Security configuration appears adequate")
        
        return recommendations
    
    def get_error_response(self, url, error_msg):
        """Return standardized error response"""
        return {
            'security_status': 'Unknown',
            'confidence': '0.00%',
            'security_score': '50.00/100',
            'risk_level': 'Medium Risk',
            'ssl_security': {'status': 'Unknown'},
            'header_security': {'status': 'Unknown'},
            'content_security': {'status': 'Unknown'},
            'recommendations': [f'Analysis failed: {error_msg}'],
            'error': error_msg
        }