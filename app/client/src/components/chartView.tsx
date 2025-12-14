import React from 'react';
import { ResultChart } from './resultChart';

interface ChartViewProps {
  yTruth: number[];
  yPredGar: number[];
  yPredGarExp: number[];
  yPredMae: number[];
  isVisible: boolean;
  onToggle: () => void;
}

export const ChartView: React.FC<ChartViewProps> = ({
  yTruth,
  yPredGar,
  yPredGarExp,
  yPredMae,
  isVisible,
  onToggle,
}) => {
  return (
    <div className="chart-view-container">
      <button className="chart-toggle-btn" onClick={onToggle}>
        {isVisible ? '📊 Hide Charts' : '📊 Show Charts'}
      </button>
      
      {isVisible && yTruth.length > 0 && (
        <div className="chart-view-content">
          <ResultChart
            yTruth={yTruth}
            yPredGar={yPredGar}
            yPredGarExp={yPredGarExp}
            yPredMae={yPredMae}
          />
        </div>
      )}
    </div>
  );
};
