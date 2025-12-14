import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

interface ResultChartProps {
  yTruth: number[];
  yPredGar: number[];
  yPredGarExp: number[];
  yPredMae: number[];
}

export const ResultChart: React.FC<ResultChartProps> = ({ yTruth, yPredGar, yPredGarExp, yPredMae }) => {
  // Combine all data into one dataset
  const chartData = yTruth.map((truth, index) => ({
    index: index + 1,
    truth,
    gar: yPredGar[index] || 0,
    garExp: yPredGarExp[index] || 0,
    mae: yPredMae[index] || 0,
  }));

  // Calculate dynamic width based on number of data points
  // 20px per data point for better spacing and visibility
  const chartWidth = chartData.length * 20;

  return (
    <div className="result-chart-container">
      <h3 className="chart-title">Prediction Comparison - All Models</h3>
      
      <div className="chart-scroll-wrapper">
        <div style={{ width: chartWidth, height: 500 }}>
          <LineChart width={chartWidth} height={500} data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
            <XAxis 
              dataKey="index" 
              label={{ value: 'Sample Index', position: 'insideBottom', offset: -5 }}
              stroke="#666"
            />
            <YAxis 
              label={{ value: 'Value', angle: -90, position: 'insideLeft' }}
              stroke="#666"
            />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: 'rgba(255, 255, 255, 0.95)',
                border: '1px solid #ddd',
                borderRadius: '8px'
              }}
            />
            <Legend />
            
            {/* Ground Truth Line */}
            <Line 
              type="monotone" 
              dataKey="truth" 
              stroke="#4a5568" 
              strokeWidth={3}
              dot={false}
              name="Ground Truth"
            />
            
            {/* GAR Prediction */} 
            <Line 
              type="monotone" 
              dataKey="gar" 
              stroke="red" 
              strokeWidth={2}
              dot={false}
              name="GAR"
            />
            
            {/* GAR-EXP Prediction */}
            <Line 
              type="monotone" 
              dataKey="garExp" 
              stroke="green" 
              strokeWidth={2}
              dot={false}
              name="GAR-EXP"
            />
            
            {/* MAE Prediction */}
            <Line 
              type="monotone" 
              dataKey="mae" 
              stroke="blue" 
              strokeWidth={2}
              dot={false}
              name="MAE"
            />
          </LineChart>
        </div>
      </div>
    </div>
  );
};
