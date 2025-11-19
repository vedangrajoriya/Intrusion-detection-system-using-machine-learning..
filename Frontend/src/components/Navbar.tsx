import { Shield, Menu, X } from 'lucide-react';
import { useState } from 'react';

interface NavbarProps {
  onNavigate: (section: string) => void;
}

const Navbar = ({ onNavigate }: NavbarProps) => {
  const [isOpen, setIsOpen] = useState(false);

  const navContainerStyle: React.CSSProperties = {
    position: 'fixed',
    top: 0,
    left: 0,
    right: 0,
    zIndex: 1000,
    background: 'rgba(15, 15, 35, 0.5)',
    backdropFilter: 'blur(10px)',
    borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
  };

  const navContentStyle: React.CSSProperties = {
    maxWidth: '1200px',
    margin: '0 auto',
    padding: '1rem 1.5rem',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
  };

  const logoStyle: React.CSSProperties = {
    display: 'flex',
    alignItems: 'center',
    gap: '0.75rem',
    color: '#ffffff',
    fontSize: '1.5rem',
    fontWeight: 'bold',
    cursor: 'pointer',
    textShadow: '0 0 10px rgba(59, 130, 246, 0.5)',
  };

  const desktopMenuStyle: React.CSSProperties = {
    display: 'flex',
    gap: '2rem',
    alignItems: 'center',
  };

  const navLinkStyle: React.CSSProperties = {
    color: '#cbd5e1',
    fontSize: '1rem',
    cursor: 'pointer',
    transition: 'all 0.3s ease',
    padding: '0.5rem 1rem',
    borderRadius: '0.5rem',
  };

  const mobileMenuButtonStyle: React.CSSProperties = {
    display: 'none',
    color: '#ffffff',
    cursor: 'pointer',
  };

  const mobileMenuStyle: React.CSSProperties = {
    display: isOpen ? 'flex' : 'none',
    flexDirection: 'column',
    gap: '1rem',
    padding: '1.5rem',
    background: 'rgba(15, 15, 35, 0.7)',
    backdropFilter: 'blur(10px)',
    borderTop: '1px solid rgba(255, 255, 255, 0.1)',
  };

  const mobileLinkStyle: React.CSSProperties = {
    color: '#cbd5e1',
    fontSize: '1.125rem',
    cursor: 'pointer',
    padding: '0.75rem',
    borderRadius: '0.5rem',
    transition: 'all 0.3s ease',
  };

  const handleNavClick = (section: string) => {
    onNavigate(section);
    setIsOpen(false);
  };

  return (
    <nav style={navContainerStyle}>
      <div style={navContentStyle}>
        <div style={logoStyle} onClick={() => handleNavClick('home')}>
          <Shield size={32} />
          <span>IDS Analyzer</span>
        </div>

        <div
          style={{
            ...desktopMenuStyle,
            display: window.innerWidth > 768 ? 'flex' : 'none',
          }}
        >
          <div
            style={navLinkStyle}
            onClick={() => handleNavClick('home')}
            onMouseEnter={(e) => {
              e.currentTarget.style.color = '#3b82f6';
              e.currentTarget.style.background = 'rgba(59, 130, 246, 0.1)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.color = '#cbd5e1';
              e.currentTarget.style.background = 'transparent';
            }}
          >
            Analyzer
          </div>
          <div
            style={navLinkStyle}
            onClick={() => handleNavClick('about')}
            onMouseEnter={(e) => {
              e.currentTarget.style.color = '#3b82f6';
              e.currentTarget.style.background = 'rgba(59, 130, 246, 0.1)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.color = '#cbd5e1';
              e.currentTarget.style.background = 'transparent';
            }}
          >
            About
          </div>
          <div
            style={navLinkStyle}
            onClick={() => handleNavClick('how-to-use')}
            onMouseEnter={(e) => {
              e.currentTarget.style.color = '#3b82f6';
              e.currentTarget.style.background = 'rgba(59, 130, 246, 0.1)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.color = '#cbd5e1';
              e.currentTarget.style.background = 'transparent';
            }}
          >
            How to Use
          </div>
        </div>

        <div
          style={{
            ...mobileMenuButtonStyle,
            display: window.innerWidth <= 768 ? 'block' : 'none',
          }}
          onClick={() => setIsOpen(!isOpen)}
        >
          {isOpen ? <X size={28} /> : <Menu size={28} />}
        </div>
      </div>

      <div style={mobileMenuStyle}>
        <div
          style={mobileLinkStyle}
          onClick={() => handleNavClick('home')}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = 'rgba(59, 130, 246, 0.1)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = 'transparent';
          }}
        >
          Analyzer
        </div>
        <div
          style={mobileLinkStyle}
          onClick={() => handleNavClick('about')}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = 'rgba(59, 130, 246, 0.1)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = 'transparent';
          }}
        >
          About
        </div>
        <div
          style={mobileLinkStyle}
          onClick={() => handleNavClick('how-to-use')}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = 'rgba(59, 130, 246, 0.1)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = 'transparent';
          }}
        >
          How to Use
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
