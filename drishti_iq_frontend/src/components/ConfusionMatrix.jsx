import React from 'react';
import { FiCheckCircle, FiXCircle } from 'react-icons/fi';

const ConfusionMatrix = ({ metrics }) => {
  if (!metrics) return null;

  // Calculate the values from the metrics file
  const support_0 = metrics['0'].support;
  const recall_0 = metrics['0'].recall;
  const true_negatives = Math.round(support_0 * recall_0);
  const false_positives = Math.round(support_0 * (1 - recall_0));

  const support_1 = metrics['1'].support;
  const recall_1 = metrics['1'].recall;
  const true_positives = Math.round(support_1 * recall_1);
  const false_negatives = Math.round(support_1 * (1 - recall_1));

  return (
    <div className="bg-slate-800 p-6 rounded-xl border border-slate-700">
      <h3 className="text-xl font-bold text-white mb-4">Ensemble Model Performance Matrix</h3>
      <div className="flex items-center">
        <div className="transform -rotate-90 whitespace-nowrap text-gray-400 font-semibold">Predicted Class</div>
        <div className="flex-grow">
          <div className="text-center text-gray-400 font-semibold mb-2">Actual Class</div>
          <div className="grid grid-cols-2 gap-px bg-slate-700 p-px rounded-lg">
            {/* True Negatives */}
            <div className="bg-slate-800 p-4 rounded-tl-lg">
              <div className="flex items-center gap-2 text-green-400"><FiCheckCircle /> <span className="font-bold">True Negative</span></div>
              <p className="text-4xl font-bold text-white my-2">{true_negatives.toLocaleString()}</p>
              <p className="text-sm text-gray-400">Normal events correctly identified.</p>
            </div>
            {/* False Positives */}
            <div className="bg-slate-800 p-4 rounded-tr-lg">
              <div className="flex items-center gap-2 text-yellow-400"><FiXCircle /> <span className="font-bold">False Positive</span></div>
              <p className="text-4xl font-bold text-white my-2">{false_positives.toLocaleString()}</p>
              <p className="text-sm text-gray-400">Normal events flagged for review.</p>
            </div>
            {/* False Negatives */}
            <div className="bg-slate-800 p-4 rounded-bl-lg">
              <div className="flex items-center gap-2 text-red-400"><FiXCircle /> <span className="font-bold">False Negative</span></div>
              <p className="text-4xl font-bold text-white my-2">{false_negatives.toLocaleString()}</p>
              <p className="text-sm text-gray-400">Anomalies the model missed.</p>
            </div>
            {/* True Positives */}
            <div className="bg-slate-800 p-4 rounded-br-lg">
              <div className="flex items-center gap-2 text-green-400"><FiCheckCircle /> <span className="font-bold">True Positive</span></div>
              <p className="text-4xl font-bold text-white my-2">{true_positives.toLocaleString()}</p>
              <p className="text-sm text-gray-400">Anomalies correctly detected.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ConfusionMatrix;