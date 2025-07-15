import { useState, useEffect } from 'react';

function App() {

  const [text, setText] = useState('');
  const [darkMode, setDarkMode] = useState(false);
  const [result, setResult] = useState<{ label: string; score: number } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    document.body.className = darkMode ? 'dark' : '';
  }, [darkMode]);

  const handleSubmit = async () => {
    
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });

      if (!res.ok) {
        throw new Error(`Server responded with status ${res.status}`);
      }

      const data = await res.json();
      setResult({ label: data.label, score: data.score });
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 dark:bg-gray-900 flex items-center justify-center p-4">
      <div className="max-w-2xl w-full bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
        <h1 className="text-2xl font-bold text-gray-800 dark:text-gray-200 mb-4">
          Electronix AI Sentiment Analysis
        </h1>
        <button
          onClick={() => setDarkMode(!darkMode)}
          className="mb-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 dark:bg-blue-700 dark:hover:bg-blue-800"
        >
          Toggle {darkMode ? 'Light' : 'Dark'} Mode
        </button>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Enter text for sentiment analysis"
          className="w-full p-3 border rounded-lg mb-4 text-gray-800 dark:text-gray-200 bg-gray-50 dark:bg-gray-700 border-gray-300 dark:border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        <button
          onClick={handleSubmit}
          className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600 dark:bg-green-700 dark:hover:bg-green-800"
        >
          Analyze
        </button>
        {loading && <p className="mt-4 text-gray-600 dark:text-gray-400">Analyzing...</p>}
        {error && <p className="mt-4 text-red-500 dark:text-red-400">Error: {error}</p>}
        {result && (
          <p className="mt-4 text-gray-800 dark:text-gray-200">
            Sentiment: {result.label} (Score: {result.score.toFixed(2)})
          </p>
        )}
      </div>
    </div>
  );
};

export default App;
