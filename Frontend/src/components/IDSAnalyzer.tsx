import { useState, useEffect, useRef } from 'react';
import {
  Shield,
  AlertTriangle,
  CheckCircle,
  Clock,
  Server,
  Lock,
  FileText,
  AlertCircle,
  Loader,
} from 'lucide-react';

interface AnalysisResult {
  security_status: string;
  confidence: string;
  security_score: string;
  risk_level: string;
  analysis_time: string;
  ssl_security: { status: string };
  header_security: { status: string };
  content_security: { status: string };
  recommendations: string[];
}

const PrismaticBurst = ({
  animationType = 'rotate3d',
  intensity = 1.5,
  speed = 0.3,
  distort = 0.8,
  rayCount = 16,
  mixBlendMode = 'lighten',
  colors = ['#3b82f6', '#1d4ed8', '#7c3aed', '#06b6d4'],
}: {
  animationType?: string;
  intensity?: number;
  speed?: number;
  distort?: number;
  rayCount?: number;
  mixBlendMode?: string;
  colors?: string[];
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const setCanvasSize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };

    setCanvasSize();
    window.addEventListener('resize', setCanvasSize);

    let rotation = 0;
    let colorOffset = 0;

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const centerX = canvas.width / 2;
      const centerY = canvas.height / 2;
      const maxRadius = Math.max(canvas.width, canvas.height) * 1.2;

      rotation += speed * 0.5;
      colorOffset += 0.005;

      for (let i = 0; i < rayCount; i++) {
        const angle = (i / rayCount) * Math.PI * 2 + rotation * 0.05;
        const nextAngle = ((i + 1) / rayCount) * Math.PI * 2 + rotation * 0.05;

        const colorIndex = Math.floor((i + colorOffset * 10) % colors.length);
        const nextColorIndex = (colorIndex + 1) % colors.length;

        const gradient = ctx.createLinearGradient(
          centerX,
          centerY,
          centerX + Math.cos(angle) * maxRadius,
          centerY + Math.sin(angle) * maxRadius
        );

        gradient.addColorStop(0, 'rgba(0, 0, 0, 0)');
        gradient.addColorStop(0.3, colors[colorIndex] + '40');
        gradient.addColorStop(0.6, colors[nextColorIndex] + '20');
        gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');

        ctx.save();
        ctx.translate(centerX, centerY);

        if (animationType === 'rotate3d') {
          const scale = Math.cos(rotation * 0.1 + i * 0.5) * 0.3 + 0.7;
          ctx.scale(1, scale);
        }

        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.lineTo(
          Math.cos(angle) * maxRadius,
          Math.sin(angle) * maxRadius
        );
        ctx.lineTo(
          Math.cos(nextAngle) * maxRadius,
          Math.sin(nextAngle) * maxRadius
        );
        ctx.closePath();

        ctx.fillStyle = gradient;
        ctx.fill();
        ctx.restore();
      }

      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener('resize', setCanvasSize);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [animationType, intensity, speed, distort, rayCount, colors]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        mixBlendMode: mixBlendMode as any,
        pointerEvents: 'none',
        zIndex: 0,
      }}
    />
  );
};

interface IDSAnalyzerProps {
  analyzerRef?: React.RefObject<HTMLDivElement>;
}

const IDSAnalyzer = ({ analyzerRef }: IDSAnalyzerProps) => {
  const [url, setUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState('');

  const getStatusIcon = (status: string) => {
    const lowerStatus = status.toLowerCase();
    if (lowerStatus === 'secure' || lowerStatus === 'good') {
      return <CheckCircle style={{ color: '#10b981', width: 24, height: 24 }} />;
    } else if (
      lowerStatus === 'needs improvement' ||
      lowerStatus === 'moderate'
    ) {
      return <AlertTriangle style={{ color: '#f59e0b', width: 24, height: 24 }} />;
    } else if (lowerStatus === 'poor' || lowerStatus === 'insecure') {
      return <AlertCircle style={{ color: '#dc2626', width: 24, height: 24 }} />;
    } else {
      return <AlertCircle style={{ color: '#6b7280', width: 24, height: 24 }} />;
    }
  };

  const getRiskColor = (riskLevel: string) => {
    const lower = riskLevel.toLowerCase();
    if (lower.includes('low')) return '#10b981';
    if (lower.includes('medium')) return '#f59e0b';
    if (lower.includes('high')) return '#dc2626';
    return '#6b7280';
  };

  const handleAnalyze = async () => {
    if (!url.trim()) return;

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const response = await fetch('http://localhost:8000/api/analyze-url/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Failed to analyze URL. Please check your connection and try again.'
      );
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !loading && url.trim()) {
      handleAnalyze();
    }
  };

  const containerStyle: React.CSSProperties = {
    minHeight: '100vh',
    background: 'transparent',
    position: 'relative',
    overflow: 'auto',
  };

  const contentOverlayStyle: React.CSSProperties = {
    position: 'relative',
    zIndex: 10,
    padding: '3rem 1.5rem',
    maxWidth: '1200px',
    margin: '0 auto',
  };

  const headerStyle: React.CSSProperties = {
    textAlign: 'center',
    marginBottom: '3rem',
  };

  const titleStyle: React.CSSProperties = {
    fontSize: '3rem',
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: '1rem',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '1rem',
    textShadow: '0 0 20px rgba(59, 130, 246, 0.5)',
  };

  const subtitleStyle: React.CSSProperties = {
    fontSize: '1.25rem',
    color: '#cbd5e1',
    textShadow: '0 2px 4px rgba(0, 0, 0, 0.3)',
  };

  const cardStyle: React.CSSProperties = {
    background: 'rgba(255, 255, 255, 0.05)',
    backdropFilter: 'blur(10px)',
    borderRadius: '1rem',
    padding: '2rem',
    boxShadow: '0 8px 32px 0 rgba(31, 38, 135, 0.37)',
    border: '1px solid rgba(255, 255, 255, 0.18)',
    marginBottom: '2rem',
  };

  const inputContainerStyle: React.CSSProperties = {
    display: 'flex',
    gap: '1rem',
    flexWrap: 'wrap',
  };

  const inputStyle: React.CSSProperties = {
    flex: 1,
    minWidth: '300px',
    padding: '1rem',
    borderRadius: '0.5rem',
    border: '1px solid rgba(255, 255, 255, 0.2)',
    background: 'rgba(255, 255, 255, 0.1)',
    color: '#ffffff',
    fontSize: '1rem',
    outline: 'none',
    transition: 'all 0.3s ease',
  };

  const buttonStyle: React.CSSProperties = {
    padding: '1rem 2rem',
    borderRadius: '0.5rem',
    border: 'none',
    background: loading
      ? 'linear-gradient(135deg, #6b7280 0%, #4b5563 100%)'
      : 'linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%)',
    color: '#ffffff',
    fontSize: '1rem',
    fontWeight: '600',
    cursor: loading || !url.trim() ? 'not-allowed' : 'pointer',
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
    transition: 'all 0.3s ease',
    opacity: loading || !url.trim() ? 0.6 : 1,
    boxShadow: '0 4px 15px rgba(59, 130, 246, 0.4)',
  };

  const gridStyle: React.CSSProperties = {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
    gap: '1.5rem',
  };

  const metricCardStyle: React.CSSProperties = {
    background: 'rgba(255, 255, 255, 0.08)',
    padding: '1.5rem',
    borderRadius: '0.75rem',
    border: '1px solid rgba(255, 255, 255, 0.1)',
    transition: 'all 0.3s ease',
  };

  const metricLabelStyle: React.CSSProperties = {
    color: '#cbd5e1',
    fontSize: '0.875rem',
    marginBottom: '0.5rem',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
  };

  const metricValueStyle: React.CSSProperties = {
    color: '#ffffff',
    fontSize: '1.5rem',
    fontWeight: 'bold',
    textShadow: '0 2px 4px rgba(0, 0, 0, 0.3)',
  };

  const sectionTitleStyle: React.CSSProperties = {
    fontSize: '1.5rem',
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: '1.5rem',
    display: 'flex',
    alignItems: 'center',
    gap: '0.75rem',
    textShadow: '0 2px 4px rgba(0, 0, 0, 0.3)',
  };

  const detailCardStyle: React.CSSProperties = {
    background: 'rgba(255, 255, 255, 0.08)',
    padding: '1.5rem',
    borderRadius: '0.75rem',
    border: '1px solid rgba(255, 255, 255, 0.1)',
    display: 'flex',
    flexDirection: 'column',
    gap: '1rem',
    transition: 'all 0.3s ease',
  };

  const detailHeaderStyle: React.CSSProperties = {
    display: 'flex',
    alignItems: 'center',
    gap: '0.75rem',
    color: '#ffffff',
    fontSize: '1.125rem',
    fontWeight: '600',
  };

  const detailStatusStyle: React.CSSProperties = {
    color: '#cbd5e1',
    fontSize: '0.875rem',
  };

  const errorStyle: React.CSSProperties = {
    background: 'rgba(220, 38, 38, 0.1)',
    border: '1px solid rgba(220, 38, 38, 0.5)',
    borderRadius: '0.5rem',
    padding: '1rem',
    color: '#fca5a5',
    display: 'flex',
    alignItems: 'center',
    gap: '0.75rem',
  };

  const recommendationItemStyle: React.CSSProperties = {
    background: 'rgba(245, 158, 11, 0.1)',
    border: '1px solid rgba(245, 158, 11, 0.3)',
    borderRadius: '0.5rem',
    padding: '1rem',
    color: '#fcd34d',
    marginBottom: '0.75rem',
    display: 'flex',
    gap: '0.75rem',
  };

  const spinnerStyle: React.CSSProperties = {
    animation: 'spin 1s linear infinite',
  };

  return (
    <div style={containerStyle}>

      <div style={contentOverlayStyle} ref={analyzerRef}>
        <div style={headerStyle}>
          <h1 style={titleStyle}>
            <Shield size={48} />
            Intrusion Detection System
          </h1>
          <p style={subtitleStyle}>
            Advanced URL Security Analysis & Threat Detection
          </p>
        </div>

        <div style={cardStyle}>
          <div style={inputContainerStyle}>
            <input
              type="text"
              placeholder="Enter URL to analyze (e.g., https://example.com)"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              onKeyPress={handleKeyPress}
              style={inputStyle}
              disabled={loading}
            />
            <button
              onClick={handleAnalyze}
              disabled={loading || !url.trim()}
              style={buttonStyle}
            >
              {loading ? (
                <>
                  <Loader size={20} style={spinnerStyle} />
                  Analyzing...
                </>
              ) : (
                <>
                  <Shield size={20} />
                  Analyze URL
                </>
              )}
            </button>
          </div>
        </div>

        {error && (
          <div style={cardStyle}>
            <div style={errorStyle}>
              <AlertCircle size={24} />
              <span>{error}</span>
            </div>
          </div>
        )}

        {result && (
          <>
            <div style={cardStyle}>
              <h2 style={sectionTitleStyle}>
                <Server size={24} />
                Security Overview
              </h2>
              <div style={gridStyle}>
                <div style={metricCardStyle}>
                  <div style={metricLabelStyle}>Security Status</div>
                  <div style={metricValueStyle}>{result.security_status}</div>
                </div>
                <div style={metricCardStyle}>
                  <div style={metricLabelStyle}>Confidence</div>
                  <div style={metricValueStyle}>{result.confidence}</div>
                </div>
                <div style={metricCardStyle}>
                  <div style={metricLabelStyle}>Security Score</div>
                  <div style={metricValueStyle}>{result.security_score}</div>
                </div>
                <div style={metricCardStyle}>
                  <div style={metricLabelStyle}>Risk Level</div>
                  <div
                    style={{
                      ...metricValueStyle,
                      color: getRiskColor(result.risk_level),
                    }}
                  >
                    {result.risk_level}
                  </div>
                </div>
              </div>
              <div
                style={{
                  marginTop: '1.5rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  color: '#cbd5e1',
                  fontSize: '0.875rem',
                }}
              >
                <Clock size={16} />
                Analysis Time: {result.analysis_time}
              </div>
            </div>

            <div style={cardStyle}>
              <h2 style={sectionTitleStyle}>
                <Lock size={24} />
                Detailed Analysis
              </h2>
              <div style={gridStyle}>
                <div style={detailCardStyle}>
                  <div style={detailHeaderStyle}>
                    <Lock size={20} />
                    SSL Security
                  </div>
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.5rem',
                    }}
                  >
                    {getStatusIcon(result.ssl_security.status)}
                    <span style={detailStatusStyle}>
                      {result.ssl_security.status}
                    </span>
                  </div>
                </div>
                <div style={detailCardStyle}>
                  <div style={detailHeaderStyle}>
                    <Server size={20} />
                    Header Security
                  </div>
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.5rem',
                    }}
                  >
                    {getStatusIcon(result.header_security.status)}
                    <span style={detailStatusStyle}>
                      {result.header_security.status}
                    </span>
                  </div>
                </div>
                <div style={detailCardStyle}>
                  <div style={detailHeaderStyle}>
                    <FileText size={20} />
                    Content Security
                  </div>
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.5rem',
                    }}
                  >
                    {getStatusIcon(result.content_security.status)}
                    <span style={detailStatusStyle}>
                      {result.content_security.status}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {result.recommendations && result.recommendations.length > 0 && (
              <div style={cardStyle}>
                <h2 style={sectionTitleStyle}>
                  <AlertTriangle size={24} />
                  Recommendations
                </h2>
                <div>
                  {result.recommendations.map((rec, index) => (
                    <div key={index} style={recommendationItemStyle}>
                      <span style={{ fontWeight: 'bold', minWidth: '1.5rem' }}>
                        {index + 1}.
                      </span>
                      <span>{rec}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default IDSAnalyzer;
