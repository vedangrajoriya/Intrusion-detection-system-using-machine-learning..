import { useRef } from 'react';
import Navbar from './components/Navbar';
import IDSAnalyzer from './components/IDSAnalyzer';
import About from './components/About';
import HowToUse from './components/HowToUse';
import Orb from './components/Orb';

function App() {
  const analyzerRef = useRef<HTMLDivElement>(null);
  const aboutRef = useRef<HTMLDivElement>(null);
  const howToUseRef = useRef<HTMLDivElement>(null);

  const handleNavigation = (section: string) => {
    let targetRef: React.RefObject<HTMLDivElement> | null = null;

    switch (section) {
      case 'home':
        targetRef = analyzerRef;
        break;
      case 'about':
        targetRef = aboutRef;
        break;
      case 'how-to-use':
        targetRef = howToUseRef;
        break;
    }

    if (targetRef?.current) {
      targetRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  const appContainerStyle: React.CSSProperties = {
    position: 'relative',
    width: '100%',
    minHeight: '100vh',
  };

  const backgroundContainerStyle: React.CSSProperties = {
    position: 'fixed',
    top: 0,
    left: 0,
    width: '100%',
    height: '100vh',
    zIndex: 0,
    pointerEvents: 'none',
  };

  const contentContainerStyle: React.CSSProperties = {
    position: 'relative',
    zIndex: 10,
    paddingTop: '5rem',
  };

  return (
    <div style={appContainerStyle}>
      <div style={backgroundContainerStyle}>
        <Orb
          hoverIntensity={0.5}
          rotateOnHover={true}
          hue={0}
          forceHoverState={false}
        />
      </div>

      <Navbar onNavigate={handleNavigation} />

      <div style={contentContainerStyle}>
        <IDSAnalyzer analyzerRef={analyzerRef} />
        <div ref={aboutRef}>
          <About />
        </div>
        <div ref={howToUseRef}>
          <HowToUse />
        </div>
      </div>
    </div>
  );
}

export default App;
