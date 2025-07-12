import React, { useState, useEffect } from 'react';
import { useQuery, gql } from '@apollo/client';
import './App.css';

const PREDICT_QUERY = gql`
  query Predict($text: String!) {
    predict(text: $text) {
      label
      score
    }
  }
`;

const App: React.FC = () => {
  const [text, setText] = useState('');
  const [darkMode, setDarkMode] = useState(false);
  const { data, loading, error, refetch } = useQuery(PREDICT_QUERY, {
    variables: { text },
    skip: !text,
    pollInterval: 1000,
  });

  useEffect(() => {
    document.body.className = darkMode ? 'dark-mode' : 'light-mode';
  }, [darkMode]);

  const handleSubmit = () => {
    refetch();
  };

  return (
    <div className="container">
      <h1>Electronix AI Sentiment Analysis</h1>
      <button onClick={() => setDarkMode(!darkMode)}>
        Toggle {darkMode ? 'Light' : 'Dark'} Mode
      </button>
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Enter text for sentiment analysis"
      />
      <button onClick={handleSubmit}>Analyze</button>
      {loading && <p>Loading...</p>}
      {error && <p>Error: {error.message}</p>}
      {data && (
        <p>
          Sentiment: {data.predict.label} (Score: {data.predict.score.toFixed(2)})
        </p>
      )}
    </div>
  );
};

export default App;