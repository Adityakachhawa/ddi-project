import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import styled, { createGlobalStyle, keyframes } from 'styled-components';
import { FiX, FiInfo, FiCode, FiBook, FiAlertTriangle, FiChevronRight, FiDatabase, FiCpu, FiShield, FiTwitter, FiTerminal, FiMenu, FiLinkedin, FiGithub, FiMail, FiUsers, FiActivity } from 'react-icons/fi';
import FunAndLearn from './component/FunAndLearn';
import axios from 'axios';
import { useRef } from 'react';
// import Prism from 'prismjs';
import 'prismjs/themes/prism-tomorrow.css';

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

// const fadeIn = keyframes`
//   from { opacity: 0; transform: translateY(20px); }
//   to { opacity: 1; transform: translateY(0); }
// `;

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
    font-size: 1.5rem;
    
    a {
      text-decoration: none;
      color: inherit;
    }
  }

  .desktop-links {
    display: flex;
    gap: 2rem;
    align-items: center;

    a {
      color: white;
      text-decoration: none;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      transition: color 0.3s ease;

      &:hover {
        color: var(--primary);
      }
    }
  }

  @media (max-width: 768px) {
    padding: 1rem;
    
    .desktop-links {
      display: none;
    }
  }
`;

const MobileMenu = styled.div`
  position: fixed;
  top: 0;
  right: 0;
  bottom: 0;
  width: 70%;
  background: rgba(15, 23, 42, 0.98);
  backdrop-filter: blur(15px);
  padding: 2rem;
  transform: translateX(${props => props.$isOpen ? '0' : '100%'});
  transition: transform 0.3s ease-in-out;
  z-index: 1001;
  display: flex;
  flex-direction: column;
  gap: 2rem;

  a {
    color: white;
    text-decoration: none;
    display: flex;
    align-items: center;
    gap: 1rem;
    font-size: 1.1rem;

    &:hover {
      color: var(--primary);
    }
  }
`;

const HamburgerButton = styled.button`
  background: none;
  border: none;
  color: white;
  font-size: 1.5rem;
  cursor: pointer;
  display: none;
  z-index: 1002;

  @media (max-width: 768px) {
    display: block;
  }
`;

// const Overlay = styled.div`
//   position: fixed;
//   top: 0;
//   left: 0;
//   right: 0;
//   bottom: 0;
//   background: rgba(0, 0, 0, 0.5);
//   backdrop-filter: blur(2px);
//   z-index: 1000;
//   display: ${props => props.isOpen ? 'block' : 'none'};
// `;
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
  background:rgba(30, 45, 77, 0.32);
  border-radius: 1rem;
  padding: 2rem;
  margin: 2rem 0;
  border: 1px solid rgba(255, 255, 255, 0.1);
  transition: transform 0.3s ease;
  position: relative; // Establish stacking context
  z-index: 2; // Ensure it’s above TestimonialsSection but below SuggestionsList

  &:hover {
    transform: translateY(-5px);
  }
`;


const DrugInput = styled.div`
  position: relative;
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  padding: 1rem;
  border-radius: 0.75rem;
  border: 2px solid ${props => props.$error ? 'var(--high-risk)' : 'var(--primary)'};
  background: #0f172a;
  min-height: 150px;
  align-items: center;
  transition: border-color 0.3s ease;
  z-index: 2;
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
  position: relative; // Establish stacking context
  z-index: 1; // Ensure it’s below SuggestionsList (z-index: 6000)
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

const NoticeBanner = styled.div`
  background: rgba(99, 102, 241, 0.1); // Subtle purple background
  border: 1px solid var(--primary);
  border-radius: 0.75rem;
  padding: 1.5rem;
  margin: 2rem 0;
  display: flex;
  align-items: center;
  gap: 1rem;
  color: var(--light);
  font-size: 1rem;
  text-align: center;
  justify-content: center;
  backdrop-filter: blur(5px);

  svg {
    color: var(--accent); // Pink accent for the icon
    width: 1.5rem;
    height: 1.5rem;
  }

  p {
    margin: 0;
    opacity: 0.9;
  }

  @media (max-width: 768px) {
    padding: 1rem;
    font-size: 0.9rem;
    flex-direction: column;
    gap: 0.5rem;
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
  

  @media (max-width: 768px) {
    padding: 1.5rem;
  }

  @media (max-width: 480px) {
    padding: 1rem;
  }
}

  .about-hero {
    text-align: center;
    margin-bottom: 3rem;
    position: relative;

    @media (max-width: 768px) {
      margin-bottom: 2rem;
    }

    
    h2 {
      font-size: 2.5rem;
      margin-bottom: 1rem;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 1rem;
    

      @media (max-width: 768px) {
        font-size: 2rem;
      }

      @media (max-width: 480px) {
        font-size: 1.75rem;
        flex-direction: column;
        gap: 0.5rem;
      }
    }


    .gradient-bar {
      height: 3px;
      width: 200px;
      background: var(--gradient);
      margin: 1rem auto;
      border-radius: 2px;

      
      @media (max-width: 480px) {
        width: 150px;
      }
    }

    .hero-subtext {
      font-size: 1.1rem;
      opacity: 0.9;
      max-width: 600px;
      margin: 0 auto;

      
      @media (max-width: 768px) {
        font-size: 1rem;
        padding: 0 1rem;
      }
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

    @media (max-width: 768px) {
      padding: 1.5rem;
      margin: 1.5rem 0;
    }

    @media (max-width: 480px) {
      padding: 1rem;
      margin: 1rem 0;
    }

  .section-icon{
    margin: 0.5rem; 
  }

    h3 {
      color: var(--primary);
      margin-bottom: 1.5rem;
      display: flex;
      align-items: center;
      gap: 1rem;

      @media (max-width: 480px) {
        font-size: 1.1rem;
      }
    }
  }

  .grid-section {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 2rem;
    margin: 3rem 0;

    
    @media (max-width: 768px) {
      grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
      gap: 1.5rem;
    }

    @media (max-width: 480px) {
      grid-template-columns: 1fr;
      gap: 1rem;
      margin: 2rem 0;
    }
  }

  .tech-card {
    padding: 2rem;
    border-radius: 1rem;
    text-align: center;
    transition: transform 0.3s ease;
    position: relative;
    overflow: hidden;

    @media (max-width: 768px) {
      padding: 1.5rem;
    }

    @media (max-width: 480px) {
      padding: 1.1rem;
    }

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

    
    @media (max-width: 480px) {
      grid-template-columns: 1fr;
    }
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

      @media (max-width: 480px) {
        width: 120px;
        height: 120px;
      }
    
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

      @media (max-width: 480px) {
        font-size: 1rem;
      }

    }

    p {
      opacity: 0.9;
      font-size: 0.9rem;

      @media (max-width: 480px) {
        font-size: 0.85rem;
      }
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

      @media (max-width: 480px) {
        font-size: 1rem;
      }
    }
  }

  .partner-logos {
    display: flex;
    justify-content: center;
    gap: 1.5rem;
    flex-wrap: wrap;

    
    @media (max-width: 480px) {
      gap: 1rem;
    }
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

    @media (max-width: 768px) {
      padding: 0.6rem 1rem;
      font-size: 0.85rem;
    }

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

  @media (max-width: 768px) {
    padding: 1.5rem;
  }

  @media (max-width: 480px) {
    padding: 1rem;
    max-width: 100%;
  }

  @media (max-width: 360px) {
    padding: 0.75rem;
  }
`;

const ModelStyles = styled.div`
  .model-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;

    @media (max-width: 768px) {
      padding: 1.5rem;
    }

    @media (max-width: 480px) {
      padding: 1rem;
    }
  }

  .data-highlight {
    text-align: center;
    padding: 2rem;
    margin: 2rem 0;
    background: rgba(99, 102, 241, 0.1);
    border-radius: 1rem;

    @media (max-width: 768px) {
      padding: 1.5rem;
    }
  }

  .data-source {
    display: flex;
    gap: 1rem;
    justify-content: center;
    flex-wrap: wrap;
    margin-top: 1rem;
  }

  .source-chip {
    padding: 0.5rem 1rem;
    border-radius: 2rem;
    background: rgba(99, 102, 241, 0.2);
    border: 1px solid var(--primary);
  }

  .risk-levels {
    display: flex;
    gap: 1rem;
    margin: 2rem 0;

    @media (max-width: 768px) {
      flex-direction: column;
    }
  }

  .risk-card {
    flex: 1;
    padding: 1.5rem;
    border-radius: 0.5rem;
    transition: transform 0.3s ease;

    &:hover {
      transform: translateY(-5px);
    }

    @media (max-width: 480px) {
      padding: 1rem;
    }
  }

  .high-risk { background: var(--high-risk); }
  .moderate-risk { background: var(--moderate-risk); }
  .low-risk { background: var(--low-risk); }

  .feature-section {
    display: flex;
    gap: 1.5rem;
    padding: 2rem;
    margin: 2rem 0;
    background: rgba(255, 255, 255, 0.03);
    border-radius: 1rem;
    align-items: center;

    @media (max-width: 768px) {
      flex-direction: column;
      text-align: center;
      padding: 1.5rem;
    }

    .section-icon {
      font-size: 2.5rem;
      flex-shrink: 0;

      @media (max-width: 480px) {
        font-size: 2rem;
      }
    }
  }

  .tech-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1.5rem;
    margin: 2rem 0;
  }

  .tech-card {
    padding: 1.5rem;
    text-align: center;
    background: rgba(99, 102, 241, 0.1);
    border-radius: 0.75rem;
  }

  @media (max-width: 768px) {
  .tech-grid {
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
  }
  .tech-card {
    padding: 1.25rem;
  }
}

/* Mobile View (480px and below) */
@media (max-width: 480px) {
  .tech-grid {
    grid-template-columns: 1fr;  /* Single column layout */
    gap: 0.75rem;
  }
  .tech-card {
    padding: 1rem;
  }
}

  .code-block {
    background: #1e1e1e;
    padding: 1.5rem;
    border-radius: 0.75rem;
    overflow-x: auto;

    @media (max-width: 480px) {
      padding: 1rem;
      font-size: 0.85rem;
    }
  }
`;
// Home Component (moved from App)
const Home = () => {
  const [drugs, setDrugs] = useState([]);
  const [drugValues, setDrugValues] = useState([]);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [suggestions, setSuggestions] = useState([]); // Combined suggestions
  const [inputValue, setInputValue] = useState('');
  const inputRef = useRef(null);

  const handleScrollToInput = () => {
    inputRef.current.scrollIntoView({ behavior: 'smooth' });
  };

  // Fetch combined suggestions (generics + synonyms)
  const fetchSuggestions = async (query) => {
    if (!query) {
      setSuggestions([]);
      return;
    }

    try {
      const genericsResponse = await axios.get(`${process.env.REACT_APP_API_URL}/all-drugs`, {
        params: { search: query, limit: 5 }
      });
      const genericSuggestions = genericsResponse.data.drugs.map(drug => ({
        display: `${drug} (Generic)`,
        value: drug,
        inputValue: drug // Display in input/drug tag
      }));

      const synonymsResponse = await axios.get(`${process.env.REACT_APP_API_URL}/synonym-suggestions/${query}`);
      const synonymSuggestions = (synonymsResponse.data.suggestions || []).map(suggestion => ({
        ...suggestion,
        inputValue: suggestion.display.split(' (')[0] // "dolo 650" from "dolo 650 (Brand of acetaminophen)"
      }));

      const combined = [...synonymSuggestions, ...genericSuggestions];
      const uniqueSuggestions = Array.from(
        new Map(combined.map(item => [item.display, item])).values()
      ).slice(0, 10);

      setSuggestions(uniqueSuggestions);
    } catch (err) {
      console.error("Failed to fetch suggestions:", err);
      setSuggestions([]);
    }
  };

  // Debounce to reduce API calls
  const debounce = (func, delay) => {
    let timeout;
    return (...args) => {
      clearTimeout(timeout);
      timeout = setTimeout(() => func(...args), delay);
    };
  };
  const debouncedFetchSuggestions = debounce(fetchSuggestions, 300);

  // Handle input change
  const handleInputChange = (e) => {
    const value = e.target.value;
    setInputValue(value); // Use inputValue, not setQuery
    debouncedFetchSuggestions(value); // Use debounced version
  };

  // Handle suggestion click
  const handleSuggestionClick = (suggestion) => {
    if (drugs.length < 2) {
      setDrugs([...drugs, suggestion.inputValue]); // Add "dolo 650" to tags
      setDrugValues([...drugValues, suggestion.value]); // Add "acetaminophen" internally
      setInputValue(''); // Clear input
      setSuggestions([]); // Hide dropdown
    }
  };

  // Handle Enter key press
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && inputValue && drugs.length < 2) {
      setDrugs([...drugs, inputValue]);
      // Try to find matching suggestion, else use inputValue
      const matchingSuggestion = suggestions.find(s => s.inputValue.toLowerCase() === inputValue.toLowerCase());
      setDrugValues([...drugValues, matchingSuggestion ? matchingSuggestion.value : inputValue]);
      setInputValue('');
      setSuggestions([]);
    }
  };
  const handleSubmit = async () => {
    setError(null);
    setLoading(true);
    try {
      const response = await axios.get(`${process.env.REACT_APP_API_URL}/predict`, {
        params: { drugA: drugValues[0], drugB: drugValues[1] } // Use drugValues
      });
      console.log('API Response:', response.data);
      const formattedData = {
        drugs: drugs, // Display original names
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
        <DrugInput $error={error}>
          {drugs.map((drug, index) => (
            <DrugTag key={index}>
              {drug}
              <button
                onClick={() => {
                  setDrugs(drugs.filter((_, i) => i !== index));
                  setDrugValues(drugValues.filter((_, i) => i !== index));
                }}
                style={{ background: 'none', border: 'none', color: 'white', cursor: 'pointer' }}
              >
                <FiX />
              </button>
            </DrugTag>
          ))}
          <input
            type="text"
            value={inputValue}
            placeholder="Enter drug name and press Enter"
            disabled={drugs.length >= 2}
            onChange={handleInputChange}
            onKeyPress={handleKeyPress}
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
            <SuggestionsList>
              {suggestions.map((suggestion, index) => (
                <SuggestionItem
                  key={index}
                  onClick={() => handleSuggestionClick(suggestion)}
                >
                  {suggestion.display}
                </SuggestionItem>
              ))}
            </SuggestionsList>
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
            <span style={{ color: results.color, marginLeft: '0.5rem' }}>
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
          <span>— Vasudev Jhawar, Pharmacist</span>
        </TestimonialCard>
      </TestimonialsSection>

      <CTASection>
        <h3>Ready to Ensure Drug Safety?</h3>
        <button onClick={handleScrollToInput}>Check Interactions Now</button>
      </CTASection>
    </>
  );
};

const SuggestionsList = styled.div`
  position: absolute;
  top: 100%;
  left: 0;
  right: 0;
  background-color: #1f2a44 !important;
  isolation: isolate;
  opacity: 1;
  border: 1px solid var(--primary);
  border-radius: 0.5rem;
  margin-top: 0.5rem;
  z-index: 6000;
  max-height: 200px;
  overflow-y: auto;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3); // Reintroduced

  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: #1f2a44;
    z-index: -1;
  }
`;

const SuggestionItem = styled.div`
  padding: 0.75rem;
  cursor: pointer;
  color: var(--light);
  background-color:rgb(6, 10, 24) !important;
  border-bottom: 1px solid rgba(99, 102, 241, 0.2);
  transition: all 0.2s ease;
  &:hover {
    background-color: #6366f1 !important;
    color: white;
  }
  &:last-child {
    border-bottom: none;
  }
`;

// App Component
function App() {
  return (
    <Router>
      <GlobalStyle />
      {/* <Navbar>
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
      </Navbar> */}
      <Navigation />
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
                      <h4>3L+ Medical Records</h4>
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
                            src={require('./assets/aditya.jpg')}
                            alt="Dr. Somdev Kachhawa"
                            className="avatar-image"
                          />
                        </div>
                        <h4>Aditya Kachhawa</h4>
                        <p>DDI.AI Founder</p>
                        <p className="bio-text">
                          B.J.S. Rampuria Jain College's Student<br />
                          BCA Part 3rd.
                        </p>
                      </div>
                      <div className="team-member">
                        <div className="hologram-avatar">
                          <img
                            src={require('./assets/somdev.jpg')}
                            alt="Aditya Kachhawa"
                            className="avatar-image"
                          />
                        </div>
                        <h4>Ketan Solanki</h4>
                        <p>DDI Innovation Ally

                        </p>
                        <p className="bio-text">
                          B.J.S. Rampuria Jain College's Student<br />
                          BCA Part 3rd.
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
              <ModelStyles>
                <div className="model-container">


                  {/* Data Section */}
                  <div className="data-highlight">
                    <h2><FiDatabase /> Trained on 300,000+ Drug Interactions</h2>
                    <p>Curated from trusted medical sources:</p>
                    <div className="data-source">
                      <span className="source-chip">Kaggle Datasets</span>
                      <span className="source-chip">DrugBank</span>
                      <span className="source-chip">FDA Reports</span>
                      <span className="source-chip">Clinical Trials</span>
                    </div>
                  </div>

                  {/* Risk Prediction */}
                  <h2><FiShield /> Interaction Risk Prediction</h2>
                  <div className="risk-levels">
                    <div className="risk-card high-risk">
                      <h4>High Risk (＞20%)</h4>
                      <p>Immediate medical attention needed</p>
                    </div>
                    <div className="risk-card moderate-risk">
                      <h4>Moderate Risk (5-20%)</h4>
                      <p>Consult your doctor</p>
                    </div>
                    <div className="risk-card low-risk">
                      <h4>Low Risk (＜5%)</h4>
                      <p>Generally safe with monitoring</p>
                    </div>
                  </div>

                  {/* How It Works */}
                  <div className="feature-section">
                    <FiDatabase className="section-icon" />
                    <div>
                      <h4>Medical Knowledge Base</h4>
                      <p>Analyzes 300,000+ historical medical records and known drug interactions</p>
                    </div>
                  </div>

                  <div className="feature-section">
                    <FiCpu className="section-icon" />
                    <div>
                      <h4>Dual AI Analysis</h4>
                      <p>Our multi-task neural network simultaneously predicts:</p>
                      <ul className="prediction-list">
                        <li>Interaction Risk Level (Low/Moderate/Severe)</li>
                        <li>Specific Interaction Mechanism</li>
                        <li>Clinical Recommendations</li>
                      </ul>
                    </div>
                  </div>

                  {/* Technical Overview */}
                  <h3><FiActivity /> Technical Overview</h3>
                  <div className="tech-grid">
                    <div className="tech-card">
                      <h4>Input Layer</h4>
                      <p>3+ Biological Features</p>
                    </div>
                    <div className="tech-card">
                      <h4>Hidden Layers</h4>
                      <p>1024 → 512 → 256 Nodes</p>
                    </div>
                    <div className="tech-card">
                      <h4>Output</h4>
                      <p>Risk Prediction Matrix</p>
                    </div>
                  </div>

                  {/* Code Insight */}
                  <h3><FiTerminal /> Core Algorithm</h3>
                  <pre className="code-block">
                    <code className="language-python">
                      {`# Drug Safety Prediction Flow
def check_drug_safety(drugA, drugB):
    # Step 1: Combine drug info
    drug_profile = combine_drug_info(
        get_targets(drugA),    # What the drugs act on
        get_enzymes(drugB)     # How the body handles them
    )
    
    # Step 2: Check risks and effects
    risk_level = predict_risk(
        drug_profile,          # Looks at the combined info
        trained_model='drug_safety_ai'  # Our smart model
    )
    
    interaction_type = predict_effect(
        drug_profile,          # Finds how they interact
        known_effects='70+ types'  # Matches to known patterns
    )
    
    # Step 3: Make it simple for people
    patient_advice = explain_results(
        risk=risk_level,       # Low, Moderate, or Severe
        effect=interaction_type  # Easy-to-read explanation
    )
    
    return {
        'risk': f"{risk_level} risk",       # Example: "Low risk"
        'effect': interaction_type,         # Example: "Higher chance of bleeding"
        'advice': patient_advice,           # Example: "Check with your doctor"
        'confidence': "95%+ accurate"       # How sure we are
    }`}
                    </code>
                  </pre>
                </div>
              </ModelStyles>
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
                <a href="https://docs.example.com" target="_blank" rel="noopener noreferrer">
                  <FiChevronRight /> Documentation
                </a>
                <a href="https://api.example.com" target="_blank" rel="noopener noreferrer">
                  <FiChevronRight /> API Reference
                </a>
                <a href="/security" target="_blank" rel="noopener noreferrer">
                  <FiChevronRight /> Security
                </a>
              </div>
            </div>

            <div className="footer-section">
              <h4>Connect</h4>
              <div className="social-links">
                <a href="https://twitter.com/example" target="_blank" rel="noopener noreferrer" aria-label="Twitter">
                  <FiTwitter />
                </a>
                <a href="https://github.com/example" target="_blank" rel="noopener noreferrer" aria-label="GitHub">
                  <FiGithub />
                </a>
                <a href="https://linkedin.com/company/example" target="_blank" rel="noopener noreferrer" aria-label="LinkedIn">
                  <FiLinkedin />
                </a>
                <a href="mailto:contact@example.com" aria-label="Email">
                  <FiMail />
                </a>
              </div>
            </div>
          </div>
          <NoticeBanner>
            <FiAlertTriangle />
            <p>This is a college project under development. It’s not perfect yet—we’re working hard to improve it!</p>
          </NoticeBanner>
          <div className="copyright">
            <p>© 2025 DDI.AI. All rights reserved.</p>
            <p>Powered by AADI-Gen AI Pharmacology Company</p>
          </div>
        </Footer>
      </Container>
    </Router>
  );
}

const Navigation = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  return (
    <>
      <Navbar>
        <h1><Link to="/">DDI.AI</Link></h1>

        {/* Desktop Links */}
        <div className="desktop-links">
          <Link to="/">
            <FiShield />Home
          </Link>
          <Link to="/about">
            <FiInfo /> About
          </Link>
          <Link to="/model">
            <FiCode /> Model
          </Link>
          <Link to="/fun-and-learn">
            <FiBook /> Fun & Learn
          </Link>
        </div>

        {/* Mobile Hamburger */}
        <HamburgerButton onClick={() => setIsMenuOpen(!isMenuOpen)}>
          {isMenuOpen ? <FiX /> : <FiMenu />}
        </HamburgerButton>
      </Navbar>
      

      {/* Mobile Menu */}
      <MobileMenu $isOpen={isMenuOpen}>
        <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
          <FiX
            style={{ cursor: 'pointer', fontSize: '1.5rem' }}
            onClick={() => setIsMenuOpen(false)} // Close button
          />
        </div>
        <Link to="/" onClick={() => setIsMenuOpen(false)}>
          <FiShield /> Home {/* Added Home button */}
        </Link>
        <Link to="/about" onClick={() => setIsMenuOpen(false)}>
          <FiInfo /> About
        </Link>
        <Link to="/model" onClick={() => setIsMenuOpen(false)}>
          <FiCode /> Model
        </Link>
        <Link to="/fun-and-learn" onClick={() => setIsMenuOpen(false)}>
          <FiBook /> Fun & Learn
        </Link>
      </MobileMenu>
    </>
  );
};

export default App;