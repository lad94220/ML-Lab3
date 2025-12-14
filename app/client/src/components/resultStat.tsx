import React from 'react';

interface ResultStatProps {
  garResult: number | null;
  garExpResult: number | null;
  maeResult: number | null;
}

export const ResultStat: React.FC<ResultStatProps> = ({ garResult, garExpResult, maeResult }) => {
  const results = [
    { label: 'GAR Loss', value: garResult, color: '#ff6b35' },
    { label: 'GAR-EXP Loss', value: garExpResult, color: '#f7931e' },
    { label: 'MAE Loss', value: maeResult, color: '#ff8c42' },
  ];

  return (
    <div className="result-stat-container">
      <h2 className="result-title">Prediction Results</h2>
      <div className="result-grid">
        {results.map((result, index) => (
          <div key={index} className="result-card" style={{ borderTopColor: result.color }}>
            <div className="result-label">{result.label}</div>
            <div className="result-value" style={{ color: result.color }}>
              {result.value !== null ? result.value.toFixed(4) : '—'}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
