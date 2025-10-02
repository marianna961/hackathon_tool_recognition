import React, { useState } from 'react';
import FileUploader from './components/FileUploader';
import ResultsViewer from './components/ResultsViewer';
import ModelInfo from './components/ModelInfo';
import './App.css';

function App() {
  const [currentView, setCurrentView] = useState('upload');
  const [results, setResults] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const handleResultsReady = (data) => {
    setResults(data);
    setCurrentView('results');
    setIsProcessing(false);
  };

  const handleNewTest = () => {
    setResults(null);
    setCurrentView('upload');
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>🛠️ Система распознавания инструментов</h1>
        <p>Автоматический учет авиационных инструментов на основе компьютерного зрения</p>
      </header>

      <main className="app-main">
        {currentView === 'upload' && (
          <>
            <ModelInfo />
            <FileUploader 
              onResultsReady={handleResultsReady}
              isProcessing={isProcessing}
              setIsProcessing={setIsProcessing}
            />
          </>
        )}

        {currentView === 'results' && results && (
          <ResultsViewer 
            results={results} 
            onNewTest={handleNewTest}
          />
        )}
      </main>

      <footer className="app-footer">
        <p>Hackathon Project • Авиационные инструменты • YOLOv8</p>
      </footer>
    </div>
  );
}

export default App;