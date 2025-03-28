// FunAndLearn.js
import React, { useState } from 'react';
import styled from 'styled-components';
import { FiBook, FiDollarSign } from 'react-icons/fi';

const Container = styled.div`
  margin: 2rem 0;
  padding: 2rem;
  background: rgba(30, 45, 77, 0.32);
  border-radius: 1rem;
`;

const Section = styled.div`
  margin-bottom: 2rem;
`;

const Title = styled.h3`
  display: flex;
  align-items: center;
  gap: 0.5rem;
`;

const Input = styled.input`
  padding: 0.5rem;
  margin-top: 0.5rem;
  border-radius: 0.5rem;
  border: 1px solid var(--primary);
  width: 100%;
`;

const Button = styled.button`
  background: var(--gradient);
  color: white;
  border: none;
  padding: 0.5rem 1rem;
  border-radius: 0.5rem;
  cursor: pointer;
  margin-top: 0.5rem;

  &:hover {
    opacity: 0.9;
  }
`;

const Result = styled.p`
  margin-top: 1rem;
  color: var(--light);
`;

const FunAndLearn = () => {
  const [randomWord, setRandomWord] = useState('');
  const [meaning, setMeaning] = useState('');
  const [amount, setAmount] = useState('');
  const [interestRate, setInterestRate] = useState('');
  const [doubleTime, setDoubleTime] = useState('');

  const vocabulary = [
    { word: 'Hello', meaning: 'नमस्ते' },
    { word: 'World', meaning: 'दुनिया' },
    { word: 'Money', meaning: 'पैसा' },
    { word: 'Friend', meaning: 'मित्र' },
    { word: 'Love', meaning: 'प्रेम' },
    { word: 'Knowledge', meaning: 'ज्ञान' },
    { word: 'Peace', meaning: 'शांति' },
    { word: 'Strength', meaning: 'शक्ति' },
    { word: 'Happiness', meaning: 'खुशी' },
    { word: 'Sadness', meaning: 'उदासी' },
    { word: 'Wisdom', meaning: 'बुद्धिमत्ता' },
    { word: 'Courage', meaning: 'साहस' },
    { word: 'Hope', meaning: 'आशा' },
    { word: 'Dream', meaning: 'सपना' },
    { word: 'Family', meaning: 'परिवार' },
    { word: 'Home', meaning: 'घर' },
    { word: 'Nature', meaning: 'प्रकृति' },
    { word: 'Beauty', meaning: 'सुंदरता' },
    { word: 'Art', meaning: 'कला' },
    { word: 'Music', meaning: 'संगीत' },
    { word: 'Dance', meaning: 'नृत्य' },
    { word: 'Food', meaning: 'खाना' },
    { word: 'Water', meaning: 'पानी' },
    { word: 'Fire', meaning: 'आग' },
    { word: 'Earth', meaning: 'पृथ्वी' },
    { word: 'Sky', meaning: 'आसमान' },
    { word: 'Star', meaning: 'तारा' },
    { word: 'Moon', meaning: 'चाँद' },
    { word: 'Sun', meaning: 'सूर्य' },
    { word: 'Light', meaning: 'रोशनी' },
    { word: 'Darkness', meaning: 'अंधकार' },
    { word: 'Time', meaning: 'समय' },
    { word: 'Space', meaning: 'अंतरिक्ष' },
    { word: 'Life', meaning: 'जीवन' },
    { word: 'Death', meaning: 'मृत्यु' },
    { word: 'Journey', meaning: 'यात्रा' },
    { word: 'Adventure', meaning: 'साहसिक कार्य' },
    { word: 'Challenge', meaning: 'चुनौती' },
    { word: 'Victory', meaning: 'जीत' },
    { word: 'Defeat', meaning: 'हार' },
    { word: 'Success', meaning: 'सफलता' },
    { word: 'Failure', meaning: 'असफलता' },
    { word: 'Effort', meaning: 'प्रयास' },
    { word: 'Goal', meaning: 'लक्ष्य' },
    { word: 'Dream', meaning: 'सपना' },
    { word: 'Future', meaning: 'भविष्य' },
    { word: 'Past', meaning: 'अतीत' },
    { word: 'Present', meaning: 'वर्तमान' },
    { word: 'Friendship', meaning: 'मित्रता' },
    { word: 'Trust', meaning: 'विश्वास' },
    { word: 'Respect', meaning: 'सम्मान' },
    { word: 'Kindness', meaning: 'दयालुता' },
    { word: 'Generosity', meaning: 'उदारता' },
    { word: 'Honesty', meaning: 'ईमानदारी' },
    { word: 'Integrity', meaning: 'अखंडता' },
    { word: 'Loyalty', meaning: 'निष्ठा' },
    { word: 'Patience', meaning: 'धैर्य' },
    { word: 'Gratitude', meaning: 'आभार' },
    { word: 'Forgiveness', meaning: 'क्षमाशीलता' },
    { word: 'Compassion', meaning: 'करुणा' },
    { word: 'Empathy', meaning: 'सहानुभूति' },
    { word: 'Understanding', meaning: 'समझ' },
    { word: 'Communication', meaning: 'संचार' },
    { word: 'Connection', meaning: 'संयोग' },
    { word: 'Collaboration', meaning: 'सहयोग' },
    { word: 'Teamwork', meaning: 'टीमवर्क' },
  { word: 'Leadership', meaning: 'नेतृत्व' },
  { word: 'Motivation', meaning: 'प्रेरणा' },
  { word: 'Inspiration', meaning: 'प्रेरणा' },
  { word: 'Creativity', meaning: 'रचनात्मकता' },
  { word: 'Innovation', meaning: 'नवाचार' },
  { word: 'Technology', meaning: 'प्रौद्योगिकी' },
  { word: 'Science', meaning: 'विज्ञान' },
  { word: 'Education', meaning: 'शिक्षा' },
  { word: 'Learning', meaning: 'सीखना' },
  { word: 'Knowledge', meaning: 'ज्ञान' },
  { word: 'Skill', meaning: 'कौशल' },
  { word: 'Talent', meaning: 'प्रतिभा' },
  { word: 'Ability', meaning: 'क्षमता' },
  { word: 'Potential', meaning: 'संभावना' },
  { word: 'Experience', meaning: 'अनुभव' },
  { word: 'Practice', meaning: 'अभ्यास' },
  { word: 'Discipline', meaning: 'अनुशासन' },
  { word: 'Focus', meaning: 'ध्यान' },
  { word: 'Concentration', meaning: 'एकाग्रता' },
  { word: 'Attention', meaning: 'ध्यान' },
  { word: 'Memory', meaning: 'याददाश्त' },
  { word: 'Intelligence', meaning: 'बुद्धिमत्ता' },
  { word: 'Wisdom', meaning: 'ज्ञान' },
  { word: 'Understanding', meaning: 'समझ' },
  { word: 'Insight', meaning: 'अवबोधन' },
  { word: 'Perspective', meaning: 'दृष्टिकोण' },
  { word: 'Vision', meaning: 'दृष्टि' },
  { word: 'Goal', meaning: 'लक्ष्य' },
  { word: 'Ambition', meaning: 'महत्वाकांक्षा' },
  { word: 'Dream', meaning: 'सपना' },
  { word: 'Aspiration', meaning: 'इच्छा' },
  { word: 'Desire', meaning: 'इच्छा' },
  { word: 'Passion', meaning: 'जुनून' },
  { word: 'Enthusiasm', meaning: 'उत्साह' },
  { word: 'Joy', meaning: 'आनंद' },
  { word: 'Sorrow', meaning: 'दुख' },
  { word: 'Fear', meaning: 'डर' },
  { word: 'Anger', meaning: 'गुस्सा' },
  { word: 'Surprise', meaning: 'आश्चर्य' },
  { word: 'Excitement', meaning: 'उत्साह' },
  { word: 'Curiosity', meaning: 'जिज्ञासा' },
  { word: 'Wonder', meaning: 'आश्चर्य' },
  { word: 'Adventure', meaning: 'साहसिक कार्य' },
  { word: 'Exploration', meaning: 'अन्वेषण' },
  { word: 'Discovery', meaning: 'खोज' },
  { word: 'Journey', meaning: 'यात्रा' },
  { word: 'Travel', meaning: 'यात्रा' },
  { word: 'Destination', meaning: 'गंतव्य' },
  { word: 'Path', meaning: 'मार्ग' },
  { word: 'Road', meaning: 'सड़क' },
  { word: 'Map', meaning: 'नक्शा' },
  { word: 'Guide', meaning: 'मार्गदर्शक' },
  { word: 'Compass', meaning: 'कंपास' },
  { word: 'Adventure', meaning: 'साहसिक कार्य' },
  { word: 'Challenge', meaning: 'चुनौती' },
  { word: 'Obstacle', meaning: 'अवरोध' },
  { word: 'Barrier', meaning: 'बाधा' },
  { word: 'Solution', meaning: 'समाधान' },
  { word: 'Answer', meaning: 'उत्तर' },
  { word: 'Question', meaning: 'प्रश्न' },
  { word: 'Inquiry', meaning: 'जांच' },
  { word: 'Research', meaning: 'अनुसंधान' },
  { word: 'Experiment', meaning: 'प्रयोग' },
  { word: 'Analysis', meaning: 'विश्लेषण' },
  { word: 'Data', meaning: 'डेटा' },
  { word: 'Information', meaning: 'जानकारी' },
  { word: 'Fact', meaning: 'तथ्य' },
  { word: 'Theory', meaning: 'सिद्धांत' },
  { word: 'Hypothesis', meaning: 'परिकल्पना' },
  { word: 'Conclusion', meaning: 'निष्कर्ष' },
  { word: 'Result', meaning: 'परिणाम' },
  { word: 'Outcome', meaning: 'परिणाम' },
  { word: 'Finding', meaning: 'खोज' },
  { word: 'Study', meaning: 'अध्ययन' },
  { word: 'Subject', meaning: 'विषय' },
  { word: 'Field', meaning: 'क्षेत्र' },
  { word: 'Discipline', meaning: 'अनुशासन' },
  { word: 'Specialization', meaning: 'विशेषज्ञता' },
  { word: 'Career', meaning: 'करियर' },
  { word: 'Profession', meaning: 'पेशे' },
  { word: 'Job', meaning: 'नौकरी' },
  { word: 'Work', meaning: 'काम' },
  { word: 'Task', meaning: 'कार्य' },
  { word: 'Project', meaning: 'परियोजना' },
  { word: 'Assignment', meaning: 'असाइनमेंट' },
  { word: 'Deadline', meaning: 'अंतिम तिथि' },
  { word: 'Goal', meaning: 'लक्ष्य' },
  { word: 'Objective', meaning: 'उद्देश्य' },
  { word: 'Plan', meaning: 'योजना' },
  { word: 'Strategy', meaning: 'रणनीति' },
  { word: 'Tactic', meaning: 'कौशल' },
  { word: 'Execution', meaning: 'निष्पादन' },
  { word: 'Performance', meaning: 'प्रदर्शन' },
  { word: 'Evaluation', meaning: 'मूल्यांकन' },
  { word: 'Feedback', meaning: 'प्रतिक्रिया' },
  { word: 'Improvement', meaning: 'सुधार' },
  { word: 'Growth', meaning: 'विकास' },
  { word: 'Development', meaning: 'विकास' },
  { word: 'Change', meaning: 'परिवर्तन' },
  { word: 'Transformation', meaning: 'परिवर्तन' },
  { word: 'Adaptation', meaning: 'अनुकूलन' },
  { word: 'Innovation', meaning: 'नवाचार' },
  { word: 'Creativity', meaning: 'रचनात्मकता' },
  { word: 'Invention', meaning: 'आविष्कार' },
  { word: 'Discovery', meaning: 'खोज' },
  { word: 'Breakthrough', meaning: 'महत्वपूर्ण खोज' },
  { word: 'Advancement', meaning: 'उन्नति' },
  { word: 'Progress', meaning: 'प्रगति' },
  { word: 'Success', meaning: 'सफलता' },
  { word: 'Achievement', meaning: 'उपलब्धि' },
  { word: 'Accomplishment', meaning: 'सिद्धि' },
  { word: 'Victory', meaning: 'जीत' },
  { word: 'Triumph', meaning: 'विजय' },
  { word: 'Celebration', meaning: 'उत्सव' },
  { word: 'Festival', meaning: 'त्यौहार' },
  { word: 'Tradition', meaning: 'परंपरा' },
  { word: 'Culture', meaning: 'संस्कृति' },
  { word: 'Heritage', meaning: 'विरासत' },
  { word: 'Community', meaning: 'समुदाय' },
  { word: 'Society', meaning: 'समाज' },
  { word: 'Nation', meaning: 'राष्ट्र' },
  { word: 'World', meaning: 'दुनिया' },
  { word: 'Global', meaning: 'वैश्विक' },
  { word: 'Environment', meaning: 'पर्यावरण' },
  { word: 'Nature', meaning: 'प्रकृति' },
  { word: 'Ecosystem', meaning: 'पारिस्थितिकी तंत्र' },
  { word: 'Biodiversity', meaning: 'जैव विविधता' },
  { word: 'Sustainability', meaning: 'सततता' },
  { word: 'Conservation', meaning: 'संरक्षण' },
  { word: 'Pollution', meaning: 'प्रदूषण' },
  { word: 'Climate', meaning: 'जलवायु' },
  { word: 'Weather', meaning: 'मौसम' },
  { word: 'Resource', meaning: 'संसाधन' },
  { word: 'Energy', meaning: 'ऊर्जा' },
  { word: 'Renewable', meaning: 'नवीकरणीय' },
  { word: 'Technology', meaning: 'प्रौद्योगिकी' },
  { word: 'Innovation', meaning: 'नवाचार' },
  { word: 'Research', meaning: 'अनुसंधान' },
  { word: 'Experiment', meaning: 'प्रयोग' },
  { word: 'Data', meaning: 'डेटा' },
  { word: 'Information', meaning: 'जानकारी' },
  { word: 'Knowledge', meaning: 'ज्ञान' },
  { word: 'Skill', meaning: 'कौशल' },
  { word: 'Talent', meaning: 'प्रतिभा' },
  { word: 'Ability', meaning: 'क्षमता' },
  { word: 'Potential', meaning: 'संभावना' },
  { word: 'Experience', meaning: 'अनुभव' },
  { word: 'Practice', meaning: 'अभ्यास' },
  { word: 'Discipline', meaning: 'अनुशासन' },
  { word: 'Focus', meaning: 'ध्यान' },
  { word: 'Concentration', meaning: 'एकाग्रता' },
  { word: 'Attention', meaning: 'ध्यान' },
  { word: 'Memory', meaning: 'याददाश्त' },
  { word: 'Intelligence', meaning: 'बुद्धिमत्ता' },
  { word: 'Wisdom', meaning: 'ज्ञान' },
  { word: 'Understanding', meaning: 'समझ' },
  { word: 'Insight', meaning: 'अवबोधन' },
  { word: 'Perspective', meaning: 'दृष्टिकोण' },
  { word: 'Vision', meaning: 'दृष्टि' },
  { word: 'Goal', meaning: 'लक्ष्य' },
  { word: 'Ambition', meaning: 'महत्वाकांक्षा' },
  { word: 'Dream', meaning: 'सपना' },
  { word: 'Aspiration', meaning: 'इच्छा' },
  { word: 'Desire', meaning: 'इच्छा' },
  { word: 'Passion', meaning: 'जुनून' },
  { word: 'Enthusiasm', meaning: 'उत्साह' },
  { word: 'Joy', meaning: 'आनंद' },
  { word: 'Sorrow', meaning: 'दुख' },
  { word: 'Fear', meaning: 'डर' },
  { word: 'Anger', meaning: 'गुस्सा' },
  { word: 'Surprise', meaning: 'आश्चर्य' },
  { word: 'Excitement', meaning: 'उत्साह' },
  { word: 'Curiosity', meaning: 'जिज्ञासा' },
  { word: 'Wonder', meaning: 'आश्चर्य' },
  { word: 'Adventure', meaning: 'साहसिक कार्य' },
  { word: 'Exploration', meaning: 'अन्वेषण' },
  { word: 'Discovery', meaning: 'खोज' },
  { word: 'Journey', meaning: 'यात्रा' },
  { word: 'Travel', meaning: 'यात्रा' },
  { word: 'Destination', meaning: 'गंतव्य' },
  { word: 'Path', meaning: 'मार्ग' },
  { word: 'Road', meaning: 'सड़क' },
  { word: 'Map', meaning: 'नक्शा' },
  { word: 'Guide', meaning: 'मार्गदर्शक' },
  { word: 'Compass', meaning: 'कंपास' },
  { word: 'Challenge', meaning: 'चुनौती' },
  { word: 'Obstacle', meaning: 'अवरोध' },
  { word: 'Barrier', meaning: 'बाधा' },

  ];

  const getRandomWord = () => {
    const randomIndex = Math.floor(Math.random() * vocabulary.length);
    const selectedWord = vocabulary[randomIndex];
    setRandomWord(selectedWord.word);
    setMeaning(selectedWord.meaning);
  };

  const handleCompoundInterest = () => {
    const principal = parseFloat(amount);
    const rate = parseFloat(interestRate) / 100;
    if (principal > 0 && rate > 0) {
      const time = Math.log(2) / Math.log(1 + rate);
      setDoubleTime(`Your money will double in approximately ${time.toFixed(2)} years.`);
    } else {
      setDoubleTime('Please enter valid values.');
    }
  };

  return (
    <Container>
      <Section>
        <Title><FiBook /> Vocabulary Builder</Title>
        <Button onClick={getRandomWord}>Get Random Word</Button>
        {randomWord && <Result>Word: {randomWord} - Meaning: {meaning}</Result>}
      </Section>

      <Section>
        <Title><FiDollarSign /> Compound Interest Calculator</Title>
        <Input
          type="number"
          placeholder="Enter amount"
          value={amount}
          onChange={(e) => setAmount(e.target.value)}
        />
        <Input
          type="number"
          placeholder="Enter interest rate (%)"
          value={interestRate}
          onChange={(e) => setInterestRate(e.target.value)}
        />
        <Button onClick={handleCompoundInterest}>Calculate</Button>
        {doubleTime && <Result>{doubleTime}</Result>}
      </Section>
    </Container>
  );
};

export default FunAndLearn;