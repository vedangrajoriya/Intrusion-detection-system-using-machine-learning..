import {
  Clipboard,
  Search,
  FileText,
  CheckCircle,
  ArrowRight,
} from 'lucide-react';

const HowToUse = () => {
  const sectionStyle: React.CSSProperties = {
    padding: '4rem 1.5rem',
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
    maxWidth: '700px',
    margin: '0 auto',
  };

  const stepsContainerStyle: React.CSSProperties = {
    display: 'flex',
    flexDirection: 'column',
    gap: '2rem',
    marginBottom: '4rem',
  };

  const stepCardStyle: React.CSSProperties = {
    background: 'rgba(255, 255, 255, 0.03)',
    backdropFilter: 'blur(10px)',
    borderRadius: '1rem',
    padding: '2rem',
    border: '1px solid rgba(255, 255, 255, 0.1)',
    display: 'flex',
    gap: '2rem',
    alignItems: 'flex-start',
    transition: 'all 0.3s ease',
  };

  const stepNumberStyle: React.CSSProperties = {
    minWidth: '64px',
    height: '64px',
    borderRadius: '50%',
    background: 'linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '1.5rem',
    fontWeight: 'bold',
    color: '#ffffff',
    boxShadow: '0 4px 15px rgba(59, 130, 246, 0.4)',
  };

  const stepContentStyle: React.CSSProperties = {
    flex: 1,
  };

  const stepTitleStyle: React.CSSProperties = {
    fontSize: '1.5rem',
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: '1rem',
    display: 'flex',
    alignItems: 'center',
    gap: '0.75rem',
  };

  const stepDescStyle: React.CSSProperties = {
    color: '#cbd5e1',
    lineHeight: '1.8',
    fontSize: '1rem',
    marginBottom: '1rem',
  };

  const codeBlockStyle: React.CSSProperties = {
    background: 'rgba(0, 0, 0, 0.3)',
    borderRadius: '0.5rem',
    padding: '1rem',
    fontFamily: 'monospace',
    color: '#06b6d4',
    fontSize: '0.875rem',
    overflowX: 'auto',
    border: '1px solid rgba(6, 182, 212, 0.3)',
  };

  const tipsContainerStyle: React.CSSProperties = {
    background: 'rgba(245, 158, 11, 0.05)',
    backdropFilter: 'blur(10px)',
    borderRadius: '1rem',
    padding: '2rem',
    border: '1px solid rgba(245, 158, 11, 0.3)',
  };

  const tipsTitleStyle: React.CSSProperties = {
    fontSize: '1.5rem',
    fontWeight: 'bold',
    color: '#fbbf24',
    marginBottom: '1.5rem',
    display: 'flex',
    alignItems: 'center',
    gap: '0.75rem',
  };

  const tipsListStyle: React.CSSProperties = {
    display: 'flex',
    flexDirection: 'column',
    gap: '1rem',
  };

  const tipItemStyle: React.CSSProperties = {
    color: '#fcd34d',
    display: 'flex',
    gap: '0.75rem',
    alignItems: 'flex-start',
  };

  const steps = [
    {
      icon: <Clipboard size={24} />,
      title: 'Enter the URL',
      description:
        'Copy and paste the URL you want to analyze into the input field. Make sure to include the full URL including the protocol (http:// or https://).',
      example: 'https://example.com',
    },
    {
      icon: <Search size={24} />,
      title: 'Initiate Analysis',
      description:
        'Click the "Analyze URL" button or press Enter to start the security scan. Our system will immediately begin examining the target URL.',
      example: null,
    },
    {
      icon: <FileText size={24} />,
      title: 'Review Results',
      description:
        'Wait a few seconds for the analysis to complete. You will receive a comprehensive report including security scores, risk levels, and detailed findings.',
      example: null,
    },
    {
      icon: <CheckCircle size={24} />,
      title: 'Take Action',
      description:
        'Review the recommendations provided and implement the suggested security improvements to enhance your website protection.',
      example: null,
    },
  ];

  const tips = [
    'Always use the full URL including the protocol (https:// or http://)',
    'For best results, analyze production URLs rather than development environments',
    'Run analyses periodically to catch new vulnerabilities as they emerge',
    'Pay special attention to high and medium risk findings in the report',
    'Implement recommended security headers to improve your security score',
    'SSL/TLS certificates should always show as "Secure" for production sites',
  ];

  return (
    <section style={sectionStyle} id="how-to-use">
      <div style={headerStyle}>
        <h2 style={titleStyle}>How to Use</h2>
        <p style={subtitleStyle}>
          Follow these simple steps to analyze any URL and get comprehensive
          security insights in seconds.
        </p>
      </div>

      <div style={stepsContainerStyle}>
        {steps.map((step, index) => (
          <div
            key={index}
            style={stepCardStyle}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = 'translateX(8px)';
              e.currentTarget.style.boxShadow =
                '0 8px 24px rgba(59, 130, 246, 0.2)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = 'translateX(0)';
              e.currentTarget.style.boxShadow = 'none';
            }}
          >
            <div style={stepNumberStyle}>{index + 1}</div>
            <div style={stepContentStyle}>
              <h3 style={stepTitleStyle}>
                {step.icon}
                {step.title}
              </h3>
              <p style={stepDescStyle}>{step.description}</p>
              {step.example && (
                <div style={codeBlockStyle}>{step.example}</div>
              )}
            </div>
          </div>
        ))}
      </div>

      <div style={tipsContainerStyle}>
        <h3 style={tipsTitleStyle}>
          <ArrowRight size={24} />
          Pro Tips
        </h3>
        <div style={tipsListStyle}>
          {tips.map((tip, index) => (
            <div key={index} style={tipItemStyle}>
              <CheckCircle
                size={20}
                style={{ minWidth: '20px', marginTop: '2px' }}
              />
              <span>{tip}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default HowToUse;
