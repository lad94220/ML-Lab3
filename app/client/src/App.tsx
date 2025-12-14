import { useState, useEffect } from 'react';
import { InputForm } from './components/input';
import { Result } from './components/result';
import { ChartView } from './components/chartView';

function App() {
  const [garResult, setGarResult] = useState<number | null>(null);
  const [garExpResult, setGarExpResult] = useState<number | null>(null);
  const [maeResult, setMaeResult] = useState<number | null>(null);
  const [showResults, setShowResults] = useState(false);
  
  // Chart data (fetched once on mount)
  const [yTruth, setYTruth] = useState<number[]>([]);
  const [yPredGar, setYPredGar] = useState<number[]>([]);
  const [yPredGarExp, setYPredGarExp] = useState<number[]>([]);
  const [yPredMae, setYPredMae] = useState<number[]>([]);
  const [showCharts, setShowCharts] = useState(false);
  const [chartsLoading, setChartsLoading] = useState(true);

  // Fetch chart data once on mount
  useEffect(() => {
    const fetchChartData = async () => {
      try {
        const response = await fetch('http://localhost:8000/chart-data');
        if (!response.ok) {
          throw new Error('Failed to fetch chart data');
        }
        const data = await response.json();
        
        // Extract arrays from response
        const garArrays = data.arrays.GAR;
        const garExpArrays = data.arrays['GAR-EXP'];
        const maeArrays = data.arrays.MAE;
        
        // Set chart data (use GAR's y_truth as they're all the same)
        setYTruth(garArrays.y_truth);
        setYPredGar(garArrays.y_pred);
        setYPredGarExp(garExpArrays.y_pred);
        setYPredMae(maeArrays.y_pred);
        
        console.log('Chart data loaded:', data.arrays.GAR.y_truth.length, 'samples');
      } catch (error) {
        console.error('Error fetching chart data:', error);
      } finally {
        setChartsLoading(false);
      }
    };

    fetchChartData();
  }, []);

  const handlePrediction = (gar: number, garExp: number, mae: number) => {
    setGarResult(gar);
    setGarExpResult(garExp);
    setMaeResult(mae);
    
    // Show the results modal
    setShowResults(true);
  };

  const handleCloseResults = () => {
    setShowResults(false);
  };

  const handleToggleCharts = () => {
    setShowCharts(!showCharts);
  };

  return (
    <>
      <InputForm onPredict={handlePrediction} />
      
      {/* Chart View Section */}
      {!chartsLoading && (
        <ChartView
          yTruth={yTruth}
          yPredGar={yPredGar}
          yPredGarExp={yPredGarExp}
          yPredMae={yPredMae}
          isVisible={showCharts}
          onToggle={handleToggleCharts}
        />
      )}
      
      {/* Results Modal */}
      {showResults && (
        <Result 
          garResult={garResult} 
          garExpResult={garExpResult} 
          maeResult={maeResult} 
          onClose={handleCloseResults}
        />
      )}
    </>
  )
}

export default App

