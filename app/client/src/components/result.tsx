import React from 'react';
import { ResultStat } from './resultStat';

interface ResultProps {
  garResult: number | null;
  garExpResult: number | null;
  maeResult: number | null;
  onClose: () => void;
}

export const Result: React.FC<ResultProps> = ({
  garResult,
  garExpResult,
  maeResult,
  onClose,
}) => {
  // Check if we have any results to show
  const hasResults = garResult !== null || garExpResult !== null || maeResult !== null;
  
  if (!hasResults) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>×</button>
        
        <div className="modal-body">
          {/* Statistics Section */}
          <ResultStat 
            garResult={garResult}
            garExpResult={garExpResult}
            maeResult={maeResult}
          />
        </div>
      </div>
    </div>
  );
};
