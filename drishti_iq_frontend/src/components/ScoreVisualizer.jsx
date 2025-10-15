import React from 'react';

const ScoreVisualizer = ({ score }) => {
  const numericScore = parseFloat(score);
  const widthPercentage = numericScore * 100;
  
  let bgColor = 'bg-green-500';
  if (numericScore > 0.6) bgColor = 'bg-yellow-500';
  if (numericScore > 0.8) bgColor = 'bg-red-500';

  return (
    <div className="flex items-center gap-3">
      <span className="font-mono text-white w-16">{numericScore.toFixed(4)}</span>
      <div className="w-full bg-slate-700 rounded-full h-2.5">
        <div 
          className={`${bgColor} h-2.5 rounded-full transition-all duration-500`} 
          style={{ width: `${widthPercentage}%` }}
        ></div>
      </div>
    </div>
  );
};

export default ScoreVisualizer;