import { Shield, Target, Zap, Lock, Eye, TrendingUp } from 'lucide-react';

const About = () => {
  const sectionStyle: React.CSSProperties = {
    padding: '6rem 1.5rem 4rem',
    maxWidth: '1200px',
    margin: '0 auto',
    position: 'relative',
    zIndex: 10,
  };

  const headerStyle: React.CSSProperties = {
    textAlign: 'center',
    marginBottom: '4rem',
  };

  const titleStyle: React.CSSProperties = {
    fontSize: '3rem',
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: '1rem',
    textShadow: '0 0 20px rgba(59, 130, 246, 0.5)',
  };

  const subtitleStyle: React.CSSProperties = {
    fontSize: '1.25rem',
    color: '#cbd5e1',
    maxWidth: '800px',
    margin: '0 auto',
    lineHeight: '1.8',
  };

  const featuresGridStyle: React.CSSProperties = {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
    gap: '2rem',
    marginBottom: '4rem',
  };

  const featureCardStyle: React.CSSProperties = {
    background: 'rgba(255, 255, 255, 0.03)',
    backdropFilter: 'blur(10px)',
    borderRadius: '1rem',
    padding: '2rem',
    border: '1px solid rgba(255, 255, 255, 0.1)',
    transition: 'all 0.3s ease',
  };

  const iconContainerStyle: React.CSSProperties = {
    width: '64px',
    height: '64px',
    borderRadius: '1rem',
    background: 'linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: '1.5rem',
    boxShadow: '0 4px 15px rgba(59, 130, 246, 0.4)',
  };

  const featureTitleStyle: React.CSSProperties = {
    fontSize: '1.5rem',
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: '1rem',
  };

  const featureDescStyle: React.CSSProperties = {
    color: '#cbd5e1',
    lineHeight: '1.8',
    fontSize: '1rem',
  };

  const statsContainerStyle: React.CSSProperties = {
    background: 'rgba(255, 255, 255, 0.03)',
    backdropFilter: 'blur(10px)',
    borderRadius: '1rem',
    padding: '3rem 2rem',
    border: '1px solid rgba(255, 255, 255, 0.1)',
    textAlign: 'center',
  };

  const statsTitleStyle: React.CSSProperties = {
    fontSize: '2rem',
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: '2rem',
  };

  const statsGridStyle: React.CSSProperties = {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
    gap: '2rem',
  };

  const statItemStyle: React.CSSProperties = {
    padding: '1.5rem',
  };

  const statValueStyle: React.CSSProperties = {
    fontSize: '3rem',
    fontWeight: 'bold',
    background: 'linear-gradient(135deg, #3b82f6, #06b6d4)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    backgroundClip: 'text',
    marginBottom: '0.5rem',
  };

  const statLabelStyle: React.CSSProperties = {
    color: '#cbd5e1',
    fontSize: '1rem',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
  };

  const features = [
    {
      icon: <Target size={32} />,
      title: 'Real-Time Analysis',
      description:
        'Get instant security assessments of any URL with comprehensive threat detection algorithms powered by machine learning.',
    },
    {
      icon: <Shield size={32} />,
      title: 'Multi-Layer Protection',
      description:
        'Analyze SSL certificates, HTTP headers, content security policies, and more to provide complete security coverage.',
    },
    {
      icon: <Zap size={32} />,
      title: 'Lightning Fast',
      description:
        'Advanced caching and optimized detection algorithms ensure results are delivered in milliseconds without compromising accuracy.',
    },
    {
      icon: <Lock size={32} />,
      title: 'Secure by Design',
      description:
        'All analyses are performed in isolated environments with zero data retention, ensuring complete privacy and security.',
    },
    {
      icon: <Eye size={32} />,
      title: 'Detailed Insights',
      description:
        'Receive comprehensive reports with actionable recommendations to improve your website security posture.',
    },
    {
      icon: <TrendingUp size={32} />,
      title: 'Continuous Learning',
      description:
        'Our system continuously learns from new threats and updates detection patterns to stay ahead of emerging vulnerabilities.',
    },
  ];

  return (
    <section style={sectionStyle} id="about">
      <div style={headerStyle}>
        <h2 style={titleStyle}>About IDS Analyzer</h2>
        <p style={subtitleStyle}>
          A state-of-the-art intrusion detection system designed to provide
          comprehensive security analysis for web applications. Our platform
          combines advanced threat detection algorithms with real-time analysis
          to keep your digital assets secure.
        </p>
      </div>

      <div style={featuresGridStyle}>
        {features.map((feature, index) => (
          <div
            key={index}
            style={featureCardStyle}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = 'translateY(-8px)';
              e.currentTarget.style.boxShadow =
                '0 12px 24px rgba(59, 130, 246, 0.2)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = 'translateY(0)';
              e.currentTarget.style.boxShadow = 'none';
            }}
          >
            <div style={iconContainerStyle}>{feature.icon}</div>
            <h3 style={featureTitleStyle}>{feature.title}</h3>
            <p style={featureDescStyle}>{feature.description}</p>
          </div>
        ))}
      </div>

      <div style={statsContainerStyle}>
        <h3 style={statsTitleStyle}>Trusted by Security Professionals</h3>
        <div style={statsGridStyle}>
          <div style={statItemStyle}>
            <div style={statValueStyle}>99.9%</div>
            <div style={statLabelStyle}>Accuracy Rate</div>
          </div>
          <div style={statItemStyle}>
            <div style={statValueStyle}>50K+</div>
            <div style={statLabelStyle}>URLs Analyzed</div>
          </div>
          <div style={statItemStyle}>
            <div style={statValueStyle}>1000+</div>
            <div style={statLabelStyle}>Threats Detected</div>
          </div>
          <div style={statItemStyle}>
            <div style={statValueStyle}>24/7</div>
            <div style={statLabelStyle}>Monitoring</div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default About;
