import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useNavigate } from 'react-router-dom';
import styled, { createGlobalStyle, keyframes } from 'styled-components';
import { FiX, FiInfo, FiCode, FiBook, FiAlertTriangle, FiChevronRight, FiDatabase, FiCpu, FiShield, FiTwitter, FiLinkedin, FiGithub, FiMail, FiUsers, FiActivity } from 'react-icons/fi';
import FunAndLearn from './component/FunAndLearn';
import axios from 'axios';
import { useRef } from 'react';
import Prism from 'prismjs';
import 'prismjs/themes/prism-tomorrow.css';

const API_URL = process.env.NODE_ENV === 'production'
  ? 'https://your-api-domain.com'
  : 'http://localhost:8000';


const GlobalStyle = createGlobalStyle`
  :root {
    --primary: #6366f1;
    --secondary: #8b5cf6;
    --accent: #d946ef;
    --dark: #0f172a;
    --light: #f8fafc;
    --gradient: linear-gradient(135deg, var(--primary), var(--secondary), var(--accent));
    --high-risk: #ef4444;
    --moderate-risk: #f59e0b;
    --low-risk: #10b981;
  }

  * {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
    font-family: 'Inter', sans-serif;
  }

  body {
    background: var(--dark);
    color: var(--light);
    min-height: 100vh;
  }

  @media (max-width: 768px) {
    html {
      font-size: 14px;
    }
  }
`;

const pulse = keyframes`
  0% { transform: scale(1); }
  50% { transform: scale(1.05); }
  100% { transform: scale(1); }
`;

const fadeIn = keyframes`
  from { opacity: 0; transform: translateY(20px); }
  to { opacity: 1; transform: translateY(0); }
`;

const Container = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem;
  padding-top: 6rem;

  @media (max-width: 768px) {
    padding: 1rem;
    padding-top: 5rem;
  }
`;

const Navbar = styled.nav`
  background: rgba(15, 23, 42, 0.95);
  padding: 1rem 2rem;
  backdrop-filter: blur(10px);
  position: fixed;
  width: 100%;
  top: 0;
  z-index: 1000;
  display: flex;
  justify-content: space-between;
  align-items: center;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);

  h1 {
    background: var(--gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    
    a {
      text-decoration: none;
      color: inherit;
    }
  }
`;

const HeroSection = styled.div`
  background: var(--gradient);
  padding: 4rem 2rem;
  border-radius: 1rem;
  text-align: center;
  margin-bottom: 2rem;
  color: white;

  h2 {
    font-size: 2.5rem;
    margin-bottom: 1rem;
  }

  p {
    font-size: 1.2rem;
    opacity: 0.9;
    margin-bottom: 2rem;
  }

  button {
    background: white;
    color: var(--dark);
    font-weight: 600;
    padding: 1rem 2rem;
    border-radius: 0.75rem;
    border: none;
    cursor: pointer;
    transition: transform 0.3s ease;

    &:hover {
      transform: translateY(-5px);
    }
  }
`;

const FeaturesGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1.5rem;
  margin: 2rem 0;
`;

const FeatureCard = styled.div`
  background: rgba(255, 255, 255, 0.05);
  padding: 1.5rem;
  border-radius: 1rem;
  text-align: center;
  transition: transform 0.3s ease;

  &:hover {
    transform: translateY(-5px);
  }

  svg {
    width: 2rem;
    height: 2rem;
    margin-bottom: 1rem;
    color: var(--primary);
  }

  h4 {
    margin-bottom: 0.5rem;
  }

  p {
    opacity: 0.8;
  }
`;

const InputCard = styled.div.attrs(() => ({ ref: undefined }))`
  background: rgba(255, 255, 255, 0.05);
  border-radius: 1rem;
  padding: 2rem;
  margin: 2rem 0;
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.1);
  transition: transform 0.3s ease;

  &:hover {
    transform: translateY(-5px);
  }
`;

const DrugInput = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  padding: 1rem;
  border-radius: 0.75rem;
  border: 2px solid ${props => props.error ? 'var(--high-risk)' : 'var(--primary)'};
  background: rgba(0, 0, 0, 0.3);
  min-height: 150px;
  align-items: center;
  transition: border-color 0.3s ease;
`;

const DrugTag = styled.div`
  background: var(--primary);
  padding: 0.5rem 1rem;
  border-radius: 2rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.9rem;
`;

const Button = styled.button`
  background: var(--gradient);
  color: white;
  border: none;
  padding: 1rem 2rem;
  border-radius: 0.75rem;
  font-size: 1rem;
  cursor: pointer;
  transition: all 0.3s ease;
  margin-top: 1.5rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  animation: ${pulse} 2s infinite;

  &:disabled {
    opacity: 0.7;
    cursor: not-allowed;
    animation: none;
  }
`;

const ResultCard = styled.div`
  background: rgba(255, 255, 255, 0.05);
  border-radius: 1rem;
  padding: 2rem;
  margin-top: 2rem;
  border: 2px solid ${props => {
    if (!props.risk) return 'transparent';
    if (props.risk === 'HIGH') return 'var(--high-risk)';
    if (props.risk === 'MODERATE') return 'var(--moderate-risk)';
    return 'var(--low-risk)';
  }};
`;

const TestimonialsSection = styled.div`
  background: rgba(255, 255, 255, 0.03);
  padding: 2rem;
  border-radius: 1rem;
  margin: 2rem 0;
`;

const TestimonialCard = styled.div`
  background: rgba(255, 255, 255, 0.05);
  padding: 1.5rem;
  border-radius: 1rem;
  margin: 1rem 0;

  p {
    opacity: 0.9;
    margin-bottom: 0.5rem;
  }

  span {
    font-size: 0.9rem;
    opacity: 0.7;
  }
`;

const CTASection = styled.div`
  background: var(--gradient);
  padding: 3rem 2rem;
  border-radius: 1rem;
  text-align: center;
  margin: 2rem 0;

  h3 {
    font-size: 2rem;
    margin-bottom: 1rem;
  }

  button {
    background: white;
    color: var(--dark);
    font-weight: 600;
    padding: 1rem 2rem;
    border-radius: 0.75rem;
    border: none;
    cursor: pointer;
    transition: transform 0.3s ease;

    &:hover {
      transform: translateY(-5px);
    }
  }
`;

const SuggestionsList = styled.div`
  position: absolute;
  background-color: var(--dark);
  border: 1px solid var(--primary);
  border-radius: 0.5rem;
  margin-top: 0.5rem;
  width: 100%;
  z-index: 1000;
`;

const SuggestionItem = styled.div`
  padding: 0.5rem;
  cursor: pointer;
  color: var(--light);
  border-bottom: 1px solid var(--primary);

  &:hover {
    background-color: var(--primary);
    color: white;
  }
`;

const Footer = styled.footer`
  background: rgba(15, 23, 42, 0.98);
  padding: 4rem 2rem 2rem;
  margin-top: 6rem;
  position: relative;
  backdrop-filter: blur(10px);
  border-top: 1px solid rgba(99, 102, 241, 0.2);
  background: linear-gradient(
    145deg,
    rgba(15, 23, 42, 0.95) 0%,
    rgba(8, 12, 24, 0.95) 100%
  );
  
  &:before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(
      90deg,
      var(--primary) 0%,
      var(--secondary) 50%,
      var(--accent) 100%
    );
    box-shadow: 0 0 15px rgba(99, 102, 241, 0.2);
  }

  .footer-content {
    max-width: 1200px;
    margin: 0 auto;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 2rem;
    padding-bottom: 3rem;
  }

  .footer-section {
    h4 {
      color: var(--primary);
      margin-bottom: 1.5rem;
      font-size: 1.1rem;
      text-transform: uppercase;
      letter-spacing: 1px;
    }
  }

  .footer-links {
    display: flex;
    flex-direction: column;
    gap: 0.8rem;

    a {
      color: rgba(255, 255, 255, 0.8);
      text-decoration: none;
      transition: all 0.3s ease;
      display: flex;
      align-items: center;
      gap: 0.5rem;

      &:hover {
        color: var(--primary);
        transform: translateX(5px);
      }

      svg {
        width: 1.2rem;
        height: 1.2rem;
      }
    }
  }

  .social-links {
    display: flex;
    gap: 1.5rem;
    justify-content: center;
    margin-top: 2rem;

    a {
      color: rgba(255, 255, 255, 0.8);
      transition: all 0.3s ease;
      padding: 0.5rem;
      
      &:hover {
        color: var(--primary);
        transform: translateY(-2px);
      }

      svg {
        width: 1.5rem;
        height: 1.5rem;
      }
    }
  }

  .copyright {
    text-align: center;
    margin-top: 3rem;
    padding-top: 2rem;
    border-top: 1px solid rgba(99, 102, 241, 0.1);
    font-size: 0.9rem;
    opacity: 0.7;
  }

  @media (max-width: 768px) {
    padding: 3rem 1rem;
    
    .footer-content {
      grid-template-columns: 1fr;
      text-align: center;
    }

    .footer-links {
      align-items: center;
    }

    .social-links {
      margin-top: 1.5rem;
    }
  }
`;

const AboutStyles = styled.div`
  .about-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
  }

  .about-hero {
    text-align: center;
    margin-bottom: 3rem;
    position: relative;
    
    h2 {
      font-size: 2.5rem;
      margin-bottom: 1rem;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 1rem;
    }

    .gradient-bar {
      height: 3px;
      width: 200px;
      background: var(--gradient);
      margin: 1rem auto;
      border-radius: 2px;
    }

    .hero-subtext {
      font-size: 1.1rem;
      opacity: 0.9;
      max-width: 600px;
      margin: 0 auto;
    }
  }

  .cyber-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(99, 102, 241, 0.2);
    padding: 2rem;
    border-radius: 1rem;
    margin: 2rem 0;
    position: relative;
    backdrop-filter: blur(10px);

  .section-icon{
    margin: 0.5rem; 
  }

    h3 {
      color: var(--primary);
      margin-bottom: 1.5rem;
      display: flex;
      align-items: center;
      gap: 1rem;
    }
  }

  .grid-section {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 2rem;
    margin: 3rem 0;
  }

  .tech-card {
    padding: 2rem;
    border-radius: 1rem;
    text-align: center;
    transition: transform 0.3s ease;
    position: relative;
    overflow: hidden;

    &::before {
      content: '';
      position: absolute;
      inset: 0;
      border: 1px solid transparent;
      background: inherit;
      border-radius: inherit;
      z-index: -1;
    }

    &:hover {
      transform: translateY(-5px);
    }

    &.glow-purple {
      border: 1px solid var(--primary);
      box-shadow: 0 0 15px rgba(99, 102, 241, 0.1);
    }

    // &.glow-blue {
    //   border: 1px solid rgba(79, 81, 245, 0.3);
    //   box-shadow: 0 0 15px rgba(10, 6, 241, 0.65);
    // }
    
    // &.glow-pink {
    //   border: 1px solid rgba(230, 78, 250, 0.9);
    //   box-shadow: 0 0 15px rgba(248, 15, 236, 0.6);
    // }

    &.glow-blue {
      border: 1px solid var(--secondary);
      box-shadow: 0 0 15px rgba(139, 92, 246, 0.1);  // Using --secondary
    }

    &.glow-pink {
      border: 1px solid var(--accent);
      box-shadow: 0 0 15px rgba(217, 70, 239, 0.1);  // Using --accent
    }
    

    svg {
      width: 2.5rem;
      height: 2.5rem;
      margin-bottom: 1rem;
      color: var(--primary);
    }
  }

  .team-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 2rem;
  }

  .team-member {
    text-align: center;
    padding: 1.5rem;
    border-radius: 1rem;
    background: rgba(0, 0, 0, 0.3);

     .hologram-avatar {
    width: 160px;
    height: 160px;
    margin: 0 auto 1rem;
    border-radius: 50%;
    position: relative;
    overflow: hidden;
    border: 2px solid rgba(99, 102, 241, 0.3);
    box-shadow: 0 0 20px rgba(99, 102, 241, 0.2);
    
    &::before {
      content: '';
      position: absolute;
      inset: 0;
      background: linear-gradient(
        45deg,
        rgba(99, 102, 241, 0.2) 0%,
        rgba(236, 72, 153, 0.2) 100%
      );
      z-index: 1;
      mix-blend-mode: screen;
    }

    .avatar-image {
      width: 100%;
      height: 100%;
      object-fit: cover;
      position: relative;
      filter: grayscale(20%) contrast(110%);
    }
  }

  // Add hover effect
  .team-member:hover {
    .hologram-avatar {
      box-shadow: 0 0 30px rgba(99, 102, 241, 0.4);
      transform: translateY(-5px);
      transition: all 0.3s ease;
      
      &::before {
        background: linear-gradient(
          45deg,
          rgba(99, 102, 241, 0.3) 0%,
          rgba(236, 72, 153, 0.3) 100%
        );
      }
    }
  }

    h4 {
      color: var(--primary);
      margin: 0.5rem 0;
    }

    p {
      opacity: 0.9;
      font-size: 0.9rem;
    }
  }

  .partners-section {
    margin-top: 3rem;
    text-align: center;

    h4 {
      margin-bottom: 2rem;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 1rem;
    }
  }

  .partner-logos {
    display: flex;
    justify-content: center;
    gap: 1.5rem;
    flex-wrap: wrap;
  }

  .logo-chip {
    padding: 0.8rem 1.5rem;
    border-radius: 2rem;
    background: rgba(99, 102, 241, 0.1);
    border: 1px solid rgba(99, 102, 241, 0.3);
    font-size: 0.9rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    transition: all 0.3s ease;

    &:hover {
      background: rgba(99, 102, 241, 0.2);
      transform: translateY(-2px);
    }
  }
`;

const AboutContainer = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem;
`;

// Home Component (moved from App)
const Home = () => {
  const [drugs, setDrugs] = useState([]);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [suggestions, setSuggestions] = useState([]); // State for suggestions
  const [inputValue, setInputValue] = useState(''); // i am use this line to clear the waste of taste after suggestion.
  // const navigate = useNavigate(); // Initialize useNavigate
  const inputRef = useRef(null);

  const handleScrollToInput = () => {
    inputRef.current.scrollIntoView({ behavior: 'smooth' });
  };

  // Fetch drug suggestions
  const fetchSuggestions = async (query) => {
    if (!query) {
      setSuggestions([]);
      return;
    }

    try {
      const response = await axios.get(`${API_URL}/all-drugs`, {
        params: { search: query, limit: 5 } // Limit to 5 suggestions
      });
      setSuggestions(response.data.drugs);
    } catch (err) {
      console.error("Failed to fetch suggestions:", err);
      setSuggestions([]);
    }
  };

  const handleSubmit = async () => {
    setError(null);
    setLoading(true);

    try {
      const response = await axios.get(`${API_URL}/predict`, {
        params: { drugA: drugs[0], drugB: drugs[1] }
      });

      console.log('API Response:', response.data);

      // Update validation for new response format
      if (!response.data ||
        !response.data.drugs ||
        !response.data.risk ||
        !response.data.interaction ||
        typeof response.data.risk_confidence === 'undefined' ||
        typeof response.data.interaction_confidence === 'undefined') {
        throw new Error('Invalid API response structure');
      }

      // Format the new response data
      const formattedData = {
        drugs: response.data.drugs,
        risk: response.data.risk,
        risk_confidence: response.data.risk_confidence,
        interaction: response.data.interaction,
        interaction_confidence: response.data.interaction_confidence,
        severity: response.data.severity,
        color: response.data.color
      };

      setResults(formattedData);
    } catch (err) {
      setError(err.response?.data?.error || err.message);
      setResults(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <HeroSection>
        <h2>AI-Powered Drug Safety</h2>
        <p>Instantly check for potential drug interactions and ensure patient safety.</p>
        {/* Updated button to navigate to Home */}
        <button onClick={handleScrollToInput}>Get Started</button>
      </HeroSection>

      <FeaturesGrid>
        <FeatureCard>
          <FiShield />
          <h4>Accurate Predictions</h4>
          <p>Powered by advanced AI trained on millions of medical records.</p>
        </FeatureCard>
        <FeatureCard>
          <FiDatabase />
          <h4>Comprehensive Database</h4>
          <p>Access a vast database of drug interactions and side effects.</p>
        </FeatureCard>
        <FeatureCard>
          <FiCpu />
          <h4>Advanced Analysis</h4>
          <p>Dual AI model predicts both risk level and interaction type</p>
        </FeatureCard>
      </FeaturesGrid>

      <InputCard ref={inputRef}>
        <DrugInput error={error}>
          {drugs.map((drug, index) => (
            <DrugTag key={index}>
              {drug}
              <button
                onClick={() => setDrugs(drugs.filter((_, i) => i !== index))}
                style={{ background: 'none', border: 'none', color: 'white', cursor: 'pointer' }}
              >
                <FiX />
              </button>
            </DrugTag>
          ))}
          <input
            type="text"
            value={inputValue} // for clear suggestions @aditya.
            placeholder="Enter drug name and press Enter"
            disabled={drugs.length >= 2}
            onChange={(e) => {
              setInputValue(e.target.value); // for clear waste after suggestion
              fetchSuggestions(e.target.value);
            }}
            onKeyPress={(e) => {
              if (e.key === 'Enter' && e.target.value.trim()) {
                if (drugs.length < 2) {
                  setDrugs([...drugs, e.target.value.trim().toLowerCase()]);
                  e.target.value = '';
                  setInputValue(''); // clear waste after suggestion
                  setSuggestions([]); // Clear suggestions after selection
                }
              }
            }}
            style={{
              background: 'transparent',
              border: 'none',
              color: 'white',
              flexGrow: 1,
              minWidth: '200px',
              padding: '0.5rem'
            }}
          />
          {suggestions.length > 0 && (
            <div style={{
              position: 'relative',
              backgroundColor: 'var(--dark)',
              border: '1px solid var(--primary)',
              borderRadius: '0.5rem',
              marginTop: '0.5rem',
              width: '100%',
              zIndex: 1000
            }}>
              {suggestions.map((drug, index) => (
                <div
                  key={index}
                  onClick={() => {
                    if (drugs.length < 2) {
                      setDrugs([...drugs, drug.toLowerCase()]);
                      setInputValue(''); // to clear waste of text after suggestion
                      setSuggestions([]); // Clear suggestions after selection
                    }
                  }}
                  style={{
                    padding: '0.5rem',
                    cursor: 'pointer',
                    color: 'var(--light)',
                    borderBottom: '1px solid var(--primary)',
                    ':hover': {
                      backgroundColor: 'var(--primary)',
                      color: 'white'
                    }
                  }}
                >
                  {drug}
                </div>
              ))}
            </div>
          )}
        </DrugInput>

        {drugs.length === 2 && (
          <Button onClick={handleSubmit} disabled={loading}>
            {loading ? 'Analyzing...' : 'Check Interaction'}
          </Button>
        )}

        {error && (
          <div style={{ color: 'var(--high-risk)', marginTop: '1rem' }}>
            <FiAlertTriangle /> {error}
          </div>
        )}
      </InputCard>

      {results && (
        <ResultCard risk={results.risk}>
          <h3>Interaction Analysis</h3>
          <p><strong>Drug Pair:</strong> {results.drugs.join(' + ')}</p>
          <p>
            <strong>Risk Level:</strong>
            <span style={{
              color: results.color,
              marginLeft: '0.5rem'
            }}>
              {results.risk} ({(results.risk_confidence * 100).toFixed(1)}%)
            </span>
          </p>
          <p><strong>Interaction:</strong> {results.interaction}</p>
          <p><strong>Interaction Confidence:</strong> {(results.interaction_confidence * 100).toFixed(1)}%</p>
          <p><strong>Recommendation:</strong> {results.severity}</p>
        </ResultCard>
      )}

      <TestimonialsSection>
        <h3>What Our Users Say</h3>
        <TestimonialCard>
          <p>"This tool has been a lifesaver for my practice. It's fast, accurate, and easy to use."</p>
          <span>— Dr. Somdev Kachhawa, Cardiologist</span>
        </TestimonialCard>
        <TestimonialCard>
          <p>"I love how detailed the interaction reports are. It helps me make better decisions for my patients."</p>
          <span>— Dr. Aditya Kachhawa, Pharmacist</span>
        </TestimonialCard>
      </TestimonialsSection>

      <CTASection>
        <h3>Ready to Ensure Drug Safety?</h3>
        {/* Updated button to navigate to Home */}
        <button onClick={handleScrollToInput}>Check Interactions Now</button>
      </CTASection>
    </>
  );
};
// App Component
function App() {
  return (
    <Router>
      <GlobalStyle />
      <Navbar>
        <h1><Link to="/">DDI.AI</Link></h1>
        <div style={{ display: 'flex', gap: '2rem' }}>
          <Link to="/about" style={{ color: 'white', textDecoration: 'none' }}>
            <FiInfo /> About
          </Link>
          <Link to="/model" style={{ color: 'white', textDecoration: 'none' }}>
            <FiCode /> Model
          </Link>
          <Link to="/fun-and-learn" style={{ color: 'white', textDecoration: 'none' }}>
            <FiBook /> Fun & Learn
          </Link>
        </div>
      </Navbar>

      <Container>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/about" element={
            <InputCard>
              <AboutStyles>
                <AboutContainer>
                  {/* Hero Section */}
                  <div className="about-hero">
                    <h2><FiShield /> Revolutionizing Drug Safety</h2>
                    <div className="gradient-bar"></div>
                    <p className="hero-subtext">
                      Combining cutting-edge AI with pharmaceutical expertise to prevent dangerous drug interactions
                    </p>
                  </div>

                  {/* Mission Section */}
                  <div className="cyber-card">
                    <FiCpu className="section-icon" />
                    <h3>Our Mission</h3>
                    <p>
                      To create a world where medication combinations are instantly analyzed using
                      artificial intelligence, preventing adverse reactions and saving lives through
                      real-time pharmacological insights.
                    </p>
                  </div>

                  {/* Technology Stack */}
                  <div className="grid-section">
                    <div className="tech-card glow-purple">
                      <FiDatabase />
                      <h4>10M+ Medical Records</h4>
                      <p>Analyzed for interaction patterns</p>
                    </div>
                    <div className="tech-card glow-blue">
                      <FiCode />
                      <h4>Dual Neural Networks</h4>
                      <p>For risk & mechanism prediction</p>
                    </div>
                    <div className="tech-card glow-pink">
                      <FiShield />
                      <h4>HIPAA-Compliant</h4>
                      <p>Enterprise-grade security</p>
                    </div>
                  </div>

                  {/* Team Section */}
                  <div className="cyber-card team-section">
                    <h3><FiUsers /> The Architects</h3>
                    <div className="team-grid">
                      <div className="team-member">
                        <div className="hologram-avatar">
                          <img
                            src={require('./assets/somdev.jpg')}
                            alt="Dr. Somdev Kachhawa"
                            className="avatar-image"
                          />
                        </div>
                        <h4>Dr. Somdev Kachhawa</h4>
                        <p>Chief Pharmacologist</p>
                        <p className="bio-text">
                          15+ years clinical experience<br />
                          MD, Pharmacology
                        </p>
                      </div>
                      <div className="team-member">
                        <div className="hologram-avatar">
                          <img
                            src={require('./assets/aditya.jpg')}
                            alt="Aditya Kachhawa"
                            className="avatar-image"
                          />
                        </div>
                        <h4>Aditya Kachhawa</h4>
                        <p>AI Architect</p>
                        <p className="bio-text">
                          Neural Networks Specialist<br />
                          PhD Computational Biology
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Data Sources */}
                  <div className="partners-section">
                    <h4><FiActivity /> Trusted By</h4>
                    <div className="partner-logos">
                      <div className="logo-chip">FDA</div>
                      <div className="logo-chip">WHO</div>
                      <div className="logo-chip">ClinicalTrials.gov</div>
                    </div>
                  </div>
                </AboutContainer>
              </AboutStyles>
            </InputCard>
          } />
          <Route path="/model" element={
            <InputCard>
              <h2><FiCode /> How Our AI Predicts Drug Interactions</h2>
              <h3><FiShield /> Understanding the Risks</h3>
              <p>Our system evaluates drug pairs using advanced machine learning to identify potential risks:</p>
              <div style={{ display: 'flex', gap: '1rem', margin: '1.5rem 0' }}>
                <div style={{ background: 'var(--high-risk)', padding: '1rem', borderRadius: '0.5rem' }}>
                  <h4>High Risk</h4>
                  <p>Immediate medical attention needed</p>
                </div>
                <div style={{ background: 'var(--moderate-risk)', padding: '1rem', borderRadius: '0.5rem' }}>
                  <h4>Moderate Risk</h4>
                  <p>Consult your doctor</p>
                </div>
                <div style={{ background: 'var(--low-risk)', padding: '1rem', borderRadius: '0.5rem' }}>
                  <h4>Low Risk</h4>
                  <p>Generally safe</p>
                </div>
              </div>

              <h3><FiCpu /> How It Works</h3>
              <div style={{ display: 'flex', gap: '1.5rem', padding: '2rem', background: 'rgba(255, 255, 255, 0.03)', borderRadius: '1rem', margin: '1.5rem 0', alignItems: 'center' }}>
                <FiDatabase />
                <div>
                  <h4>1. Medical Knowledge Base</h4>
                  <p>Analyzes 10M+ historical medical records and known drug interactions</p>
                </div>
              </div>

              <div style={{ display: 'flex', gap: '1.5rem', padding: '2rem', background: 'rgba(255, 255, 255, 0.03)', borderRadius: '1rem', margin: '1.5rem 0', alignItems: 'center' }}>
                <FiCpu />
                <div>
                  <h4>2. Dual AI Analysis</h4>
                  <p>Our multi-task neural network simultaneously predicts:</p>
                  <ul style={{ marginTop: '0.5rem', marginLeft: '0.9rem', opacity: 0.9 }}>
                    <li>Interaction Risk Level (Low/Moderate/Severe)</li>
                    <li>Specific Interaction Mechanism</li>
                    <li>Clinical Recommendations</li>
                  </ul>
                </div>
              </div>

              <h3>Technical Overview</h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1.5rem', padding: '2rem', background: 'var(--dark)', borderRadius: '1rem', margin: '2rem 0' }}>
                <div style={{ textAlign: 'center', padding: '1.5rem', background: 'rgba(99, 102, 241, 0.1)', borderRadius: '0.75rem' }}>
                  <h4>Input Layer</h4>
                  <p>1500+ Biological Features</p>
                </div>
                <div style={{ textAlign: 'center', padding: '1.5rem', background: 'rgba(99, 102, 241, 0.1)', borderRadius: '0.75rem' }}>
                  <h4>Hidden Layers</h4>
                  <p>1024 → 512 → 256 Nodes</p>
                </div>
                <div style={{ textAlign: 'center', padding: '1.5rem', background: 'rgba(99, 102, 241, 0.1)', borderRadius: '0.75rem' }}>
                  <h4>Output</h4>
                  <p>Risk Prediction</p>
                </div>
              </div>

              <h3>Code Insight</h3>
              <pre style={{ background: '#1e1e1e', padding: '1.5rem', borderRadius: '0.75rem', margin: '1.5rem 0', overflowX: 'auto' }}>
                <code className="language-python">
                  {`# Simplified AI Architecture
class DrugSafetyAI:
    def analyze(self, drug1, drug2):
        # 1. Extract biological features
        features = self.get_features(drug1, drug2)
        
        # 2. Neural network processing
        risk_prediction = self.neural_network(features)
        
        # 3. Generate human-readable explanation
        explanation = self.explain_results(risk_prediction)
        
        return {
            'risk': risk_prediction,
            'explanation': explanation
        }`}
                </code>
              </pre>
            </InputCard>
          } />
          <Route path="/fun-and-learn" element={<FunAndLearn />} />
        </Routes>

        <Footer>
          <div className="footer-content">
            <div className="footer-section">
              <h4>Navigation</h4>
              <div className="footer-links">
                <Link to="/about"><FiChevronRight /> About</Link>
                <Link to="/model"><FiChevronRight /> AI Model</Link>
                <Link to="/fun-and-learn"><FiChevronRight /> Fun & Learn</Link>
              </div>
            </div>

            <div className="footer-section">
              <h4>Resources</h4>
              <div className="footer-links">
                <a href="/docs"><FiChevronRight /> Documentation</a>
                <a href="/api"><FiChevronRight /> API Reference</a>
                <a href="/security"><FiChevronRight /> Security</a>
              </div>
            </div>

            <div className="footer-section">
              <h4>Connect</h4>
              <div className="social-links">
                <a href="#"><FiTwitter /></a>
                <a href="#"><FiGithub /></a>
                <a href="#"><FiLinkedin /></a>
                <a href="#"><FiMail /></a>
              </div>
            </div>
          </div>

          <div className="copyright">
            <p>© 2023 DDI.AI. All rights reserved.</p>
            <p>Powered by AADI-Gen AI Pharmacology Company</p>
          </div>
        </Footer>
      </Container>
    </Router>
  );
}

export default App;