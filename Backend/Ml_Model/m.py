import pandas as pd
import numpy as np
import gc  
import traceback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import requests
import socket
import whois
from urllib.parse import urlparse
import ssl
import datetime
import re
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from ratelimit import limits, sleep_and_retry
import threading
from concurrent.futures import ThreadPoolExecutor
import validators
from bs4 import BeautifulSoup

warnings.filterwarnings('ignore')

# Step 1: Dataset Setup
def load_dataset():
    """Load dataset with memory optimization"""
    dataset_path =r"D:\6th sem ki padhai\project 7 sem\backend\ml_model\cleaned_for_training.csv"
    
    print("Loading dataset with memory optimization...")
    
    # Read CSV in chunks and optimize data types
    chunks = []
    chunk_size = 100000  # Adjust this based on your system's memory
    
    # First, read the data types of columns
    dtypes = {
        'int64': 'int32',    # Reduce integer precision
        'float64': 'float32' # Reduce float precision
    }
    
    try:
        # Read the dataset in chunks
        for chunk in pd.read_csv(dataset_path, chunksize=chunk_size):
            # Optimize data types for numeric columns
            for column in chunk.select_dtypes(include=['int64']).columns:
                chunk[column] = chunk[column].astype('int32')
            for column in chunk.select_dtypes(include=['float64']).columns:
                chunk[column] = chunk[column].astype('float32')
            
            chunks.append(chunk)
        
        # Combine chunks
        df = pd.concat(chunks, ignore_index=True)
        
        # Free up memory
        del chunks
        import gc
        gc.collect()
        
        print("Dataset shape:", df.shape)
        print("Memory usage:", df.memory_usage().sum() / 1024**2, "MB")
        
        # Optional: Sample the dataset if it's still too large
        if df.shape[0] > 500000:  # Adjust this threshold as needed
            print("Dataset is large, using a random sample...")
            df = df.sample(n=500000, random_state=42)
            print("New dataset shape:", df.shape)
        
        return df
        
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        print("\nTrying alternative loading method...")
        
        # Alternative: Load only necessary columns
        try:
            # Read only the first row to get column names
            cols = pd.read_csv(dataset_path, nrows=1).columns
            
            # Select important columns (modify this based on your needs)
            important_cols = cols[-10:].tolist()  # Example: last 10 columns
            
            # Read only selected columns
            df = pd.read_csv(dataset_path, usecols=important_cols)
            print("Successfully loaded reduced dataset")
            print("Dataset shape:", df.shape)
            
            return df
            
        except Exception as e2:
            print(f"Alternative loading also failed: {str(e2)}")
            raise

# Step 2: Data Preprocessing
def analyze_features(df):
    """Analyze and select the most important features"""
    print("\nAnalyzing features...")
    
    # Print basic information about the dataset
    print("\nDataset Info:")
    print(df.info())
    
    # Display first few rows
    print("\nFirst few rows:")
    print(df.head())
    
    # Check for missing values
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Check for highly correlated features
    correlation_matrix = df.corr()
    high_corr_features = np.where(np.abs(correlation_matrix) > 0.9)
    high_corr_features = [(correlation_matrix.index[x], correlation_matrix.columns[y], correlation_matrix.iloc[x, y]) 
                         for x, y in zip(*high_corr_features) if x != y]
    
    print("\nHighly correlated features:")
    for feat1, feat2, corr in high_corr_features:
        print(f"{feat1} - {feat2}: {corr:.2f}")
    
    return df

def optimize_dtypes(df):
    """Reduce memory usage by optimizing data types"""
    
    print("\nOptimizing data types to reduce memory usage...")
    for col in df.columns:
        # Skip non-numeric columns
        if df[col].dtype == 'object':
            continue
            
        # Convert integers
        if df[col].dtype in ['int64', 'int32']:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        # Convert floats
        elif df[col].dtype in ['float64', 'float32']:
            df[col] = pd.to_numeric(df[col], downcast='float')
    
    return df

def preprocess_data(df):
    """Handle missing values and preprocess features with memory optimization"""
    
    # Convert to float32 for memory efficiency
    df = df.astype('float32', errors='ignore')
    
    # Get target column (last column)
    target_column = df.columns[-1]
    
    # Split features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column].copy()
    
    # Free memory
    del df
    gc.collect()
    
    # Handle missing values efficiently
    X = X.fillna(X.mean())
    
    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # Select fewer features
    k = min(10, X.shape[1])  # Select fewer features if needed
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    
    # Free more memory
    del X
    gc.collect()
    
    # Scale features
    scaler = StandardScaler()
    X_selected = scaler.fit_transform(X_selected)
    
    # Split with smaller test size
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.1, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, le, selected_features

# Step 3: Model Building
def train_xgboost(X_train, X_test, y_train, y_test):
    """Train XGBoost classifier with explicit class handling"""
    
    print("\nPreparing XGBoost training...")
    
    # Get number of unique classes
    num_classes = len(np.unique(y_train))
    print(f"Number of classes: {num_classes}")
    print(f"Unique classes: {np.unique(y_train)}")
    
    # Create model with updated parameters
    model = xgb.XGBClassifier(
        max_depth=5,
        learning_rate=0.1,
        n_estimators=200,
        subsample=0.8,
        objective='multi:softmax',  # Changed from multi:softprob to multi:softmax
        num_class=num_classes,
        tree_method='hist',
        random_state=42,
        use_label_encoder=False
    )
    
    # Train model
    model.fit(
        X_train, 
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=True
    )
    
    return model

# Step 4: Model Evaluation
def evaluate_model(model, X_test, y_test, le):
    """Evaluate model with robust error handling"""
    
    print("\nStarting model evaluation...")
    
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Convert predictions to same format as test data
        y_test = y_test.astype(int)
        y_pred = y_pred.astype(int)
        
        # Print shapes and unique values for debugging
        print(f"y_test shape: {y_test.shape}, unique values: {np.unique(y_test)}")
        print(f"y_pred shape: {y_pred.shape}, unique values: {np.unique(y_pred)}")
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Print results
        print("\nModel Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        
        # Get class names
        class_names = [str(c) for c in le.classes_]
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        print("\nDebug information:")
        print(f"y_test type: {type(y_test)}, shape: {y_test.shape}")
        print(f"y_pred type: {type(y_pred)}, shape: {y_pred.shape}")
        raise

# Step 5: Feature Importance
def plot_feature_importance(model, feature_names):
    """Plot feature importance"""
    
    print("\nPlotting feature importance...")
    
    # Get feature importance
    importance = model.feature_importances_
    
    # Sort feature importance in descending order
    indices = np.argsort(importance)[::-1]
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(importance)), importance[indices])
    plt.xticks(range(len(importance)), 
               [feature_names[i] for i in indices], 
               rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Add a configuration class for settings
class URLAnalyzerConfig:
    TIMEOUT = 10  # seconds
    MAX_RETRIES = 3
    RATE_LIMIT_CALLS = 60  # number of calls
    RATE_LIMIT_PERIOD = 60  # seconds
    MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5MB
    SSL_VERIFY = True  # Enable SSL verification for production

# Add a session manager for better request handling
class RequestManager:
    def __init__(self, config):
        self.config = config
        self.session = self._create_session()
        self._lock = threading.Lock()
    
    def _create_session(self):
        session = requests.Session()
        retry_strategy = Retry(
            total=self.config.MAX_RETRIES,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session
    
    @sleep_and_retry
    @limits(calls=URLAnalyzerConfig.RATE_LIMIT_CALLS, period=URLAnalyzerConfig.RATE_LIMIT_PERIOD)
    def make_request(self, url):
        with self._lock:
            return self.session.get(
                url,
                timeout=self.config.TIMEOUT,
                verify=self.config.SSL_VERIFY,
                allow_redirects=True,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            )

def extract_url_features(url, config=URLAnalyzerConfig()):
    """Extract advanced security features from URL"""
    try:
        if not validators.url(url):
            raise ValueError("Invalid URL format")
            
        features = {}
        parsed = urlparse(url)
        request_manager = RequestManager(config)
        domain = parsed.netloc
        
        # Advanced URL features
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
            'suspicious_tld': int(domain.split('.')[-1] in ['xyz', 'top', 'pw', 'cc', 'tk']),
            'has_suspicious_words': int(bool(re.search(
                r'login|admin|bank|secure|account|verify|confirm|update|payment|wallet|crypto',
                url.lower()
            ))),
            'char_entropy': sum(-p * np.log2(p) for p in 
                [url.count(c)/len(url) for c in set(url)] if p > 0)
        })
        
        # Security checks
        try:
            response = request_manager.make_request(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Content analysis
            features.update({
                'has_password_field': int(bool(soup.find('input', {'type': 'password'}))),
                'has_form': int(bool(soup.find('form'))),
                'external_links': len(set(a['href'] for a in soup.find_all('a', href=True)
                                    if a['href'].startswith('http'))),
                'iframe_count': len(soup.find_all('iframe')),
                'script_count': len(soup.find_all('script')),
                'hidden_elements': len(soup.find_all(style='display:none')),
                'redirect_count': len(response.history),
                'has_mixed_content': int(bool(
                    re.search(r'http:[^"\']*\.(js|css|img)', str(response.content))
                )),
                'response_time': response.elapsed.total_seconds(),
                'content_length': len(response.content),
                'status_code': response.status_code
            })
            
            # Security headers analysis
            headers = response.headers
            features.update({
                'has_hsts': int('strict-transport-security' in headers),
                'has_xframe': int('x-frame-options' in headers),
                'has_csp': int('content-security-policy' in headers),
                'has_xss_protection': int('x-xss-protection' in headers),
                'server_disclosed': int('server' in headers)
            })
            
        except Exception as e:
            # Set default values if request fails
            features.update({
                'has_password_field': 0, 'has_form': 0, 'external_links': 0,
                'iframe_count': 0, 'script_count': 0, 'hidden_elements': 0,
                'redirect_count': 0, 'has_mixed_content': 0, 'response_time': 0,
                'content_length': 0, 'status_code': 0, 'has_hsts': 0,
                'has_xframe': 0, 'has_csp': 0, 'has_xss_protection': 0,
                'server_disclosed': 0
            })
        
        # Certificate analysis
        try:
            ctx = ssl.create_default_context()
            with ctx.wrap_socket(socket.socket(), server_hostname=domain) as s:
                s.settimeout(config.TIMEOUT)
                s.connect((domain, 443))
                cert = s.getpeercert()
                
                # Parse certificate details
                features.update({
                    'cert_valid_days': (datetime.datetime.strptime(
                        cert['notAfter'], r'%b %d %H:%M:%S %Y %Z'
                    ) - datetime.datetime.now()).days,
                    'cert_is_valid': 1,
                    'cert_has_wildcard': int('*' in cert.get('subjectAltName', [''])[0][1]),
                    'cert_issuer_trusted': int(
                        any(trusted in str(cert.get('issuer', ''))
                            for trusted in ['Let\'s Encrypt', 'DigiCert', 'GeoTrust', 'Comodo'])
                    )
                })
        except Exception:
            features.update({
                'cert_valid_days': 0, 'cert_is_valid': 0,
                'cert_has_wildcard': 0, 'cert_issuer_trusted': 0
            })
            
        return pd.DataFrame([features])
        
    except Exception as e:
        print(f"Error extracting features for {url}: {str(e)}")
        return None

def predict_url_safety(model, scaler, selected_features, le, url, config=URLAnalyzerConfig()):
    """Enhanced URL safety prediction with detailed analysis"""
    try:
        if not validators.url(url):
            return {"error": "Invalid URL format"}
            
        # Extract features
        features_df = extract_url_features(url, config)
        if features_df is None:
            return {"error": "Feature extraction failed"}
        
        # Print available features for debugging
        print("\nAvailable features:", features_df.columns.tolist())
        print("Selected features:", selected_features)
        
        # Create a DataFrame with the same features as training data
        prediction_features = pd.DataFrame(0, index=[0], columns=selected_features)
        
        # Copy available features
        for feature in selected_features:
            if feature in features_df.columns:
                prediction_features[feature] = features_df[feature]
        
        # Calculate security score
        security_score = calculate_security_score(features_df)
        
        # Prepare features for prediction
        X = prediction_features.values
        X_scaled = scaler.transform(X)
        
        try:
            # Make prediction
            prediction = "Suspicious"  # Default prediction
            confidence = 0.0
            
            if model is not None:
                prediction = model.predict(X_scaled)[0]
                pred_prob = model.predict_proba(X_scaled)
                confidence = np.max(pred_prob) * 100
                if le is not None:
                    prediction = le.inverse_transform([prediction])[0]
        except Exception as pred_error:
            print(f"Prediction error: {str(pred_error)}")
            prediction = "Unable to predict"
            confidence = 0.0
        
        # Detailed security analysis
        security_analysis = analyze_security_features(features_df)
        
        return {
            'url': url,
            'prediction': str(prediction),
            'confidence': f"{confidence:.2f}%",
            'security_score': f"{security_score:.2f}/100",
            'risk_level': get_risk_level(security_score),
            'analysis_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'security_analysis': security_analysis,
            'recommendations': generate_security_recommendations(security_analysis)
        }
        
    except Exception as e:
        print(f"Error in predict_url_safety: {str(e)}")
        return {
            'url': url,
            'prediction': "Error",
            'confidence': "0.00%",
            'security_score': "0.00/100",
            'risk_level': "Unknown",
            'analysis_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'error': str(e)
        }

def calculate_security_score(features_df):
    """Calculate security score based on various security features"""
    weights = {
        'has_https': 10,
        'has_hsts': 8,
        'has_csp': 8,
        'has_xframe': 6,
        'has_xss_protection': 6,
        'cert_is_valid': 10,
        'cert_issuer_trusted': 8,
        'suspicious_tld': -10,
        'has_mixed_content': -8,
        'has_suspicious_words': -5,
        'server_disclosed': -3
    }
    
    score = 50  # Base score
    for feature, weight in weights.items():
        if feature in features_df.columns:
            score += weight * features_df[feature].iloc[0]
    
    return max(0, min(100, score))  # Clamp between 0 and 100

def get_risk_level(security_score):
    """Determine risk level based on security score"""
    if security_score >= 80:
        return "Low Risk"
    elif security_score >= 60:
        return "Medium Risk"
    elif security_score >= 40:
        return "High Risk"
    else:
        return "Critical Risk"

def analyze_security_features(features_df):
    """Provide detailed security analysis"""
    analysis = {
        'ssl_security': {
            'status': 'Secure' if features_df['has_https'].iloc[0] else 'Insecure',
            'details': []
        },
        'header_security': {
            'status': 'Good' if sum([
                features_df['has_hsts'].iloc[0],
                features_df['has_csp'].iloc[0],
                features_df['has_xframe'].iloc[0]
            ]) >= 2 else 'Needs Improvement',
            'details': []
        },
        'content_security': {
            'status': 'Warning' if features_df['has_mixed_content'].iloc[0] else 'Good',
            'details': []
        }
    }
    
    return analysis

def generate_security_recommendations(analysis):
    """Generate security recommendations based on analysis"""
    recommendations = []
    
    if analysis['ssl_security']['status'] == 'Insecure':
        recommendations.append("Enable HTTPS and obtain a valid SSL certificate")
    
    if analysis['header_security']['status'] == 'Needs Improvement':
        recommendations.append("Implement security headers (HSTS, CSP, X-Frame-Options)")
    
    if analysis['content_security']['status'] == 'Warning':
        recommendations.append("Fix mixed content issues and remove unsafe scripts")
    
    return recommendations

# Modify the main function to include URL analysis
def main():
    config = URLAnalyzerConfig()
    config.SSL_VERIFY = False
    
    try:
        # Load dataset
        df = load_dataset()
        df = analyze_features(df)
        
        # Preprocess data
        X_train, X_test, y_train, y_test, le, selected_features = preprocess_data(df)
        
        # Train model
        model = train_xgboost(X_train, X_test, y_train, y_test)
        
        # Initialize scaler
        scaler = StandardScaler()
        scaler.fit(X_train)
        
        print("\nModel ready for URL analysis")
        print("=" * 50)
        
        while True:
            url = input("\nEnter a URL to analyze (or 'quit' to exit): ")
            if url.lower() == 'quit':
                break
            
            print("\nAnalyzing URL security...")
            result = predict_url_safety(model, scaler, selected_features, le, url, config)
            
            print("\nAnalysis Results:")
            print("=" * 50)
            print(f"URL: {result.get('url', 'N/A')}")
            print(f"Security Status: {result.get('prediction', 'Unknown')}")
            print(f"Confidence: {result.get('confidence', '0.00%')}")
            print(f"Security Score: {result.get('security_score', '0.00/100')}")
            print(f"Risk Level: {result.get('risk_level', 'Unknown')}")
            print(f"Analysis Time: {result.get('analysis_time', 'N/A')}")
            
            if 'error' in result:
                print(f"\nError: {result['error']}")
            
            if 'security_analysis' in result:
                print("\nSecurity Analysis:")
                for category, details in result['security_analysis'].items():
                    print(f"\n{category.replace('_', ' ').title()}:")
                    print(f"Status: {details['status']}")
            
            if 'recommendations' in result:
                print("\nRecommendations:")
                for i, rec in enumerate(result['recommendations'], 1):
                    print(f"{i}. {rec}")
            
            print("=" * 50)
                
    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
        print("\nStack trace:")
        traceback.print_exc()

if __name__ == "__main__":
    main()