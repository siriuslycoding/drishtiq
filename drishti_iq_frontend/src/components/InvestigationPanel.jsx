import React from 'react';
import { FiX, FiUser, FiBarChart2, FiEdit, FiCheckCircle } from 'react-icons/fi';
import ScoreVisualizer from './ScoreVisualizer'; // We will reuse this component

const InvestigationPanel = ({ alert, onClose, onStatusChange }) => {
  if (!alert) return null;

  const levelText = alert.threat_level.replace(/[^a-zA-Z]/g, '').toUpperCase();
  const severityColor = levelText === 'HIGH' ? 'red' : levelText === 'MEDIUM' ? 'yellow' : 'blue';

  const features = [
    { name: alert.top_feature_1, value: alert.feature_1_value, importance: alert.feature_1_importance },
    { name: alert.top_feature_2, value: alert.feature_2_value, importance: alert.feature_2_importance },
    { name: alert.top_feature_3, value: alert.feature_3_value, importance: alert.feature_3_importance },
  ].filter(f => f.name);

  return (
    <div 
      className={`fixed top-0 right-0 h-full w-full md:w-1/2 lg:w-1/3 bg-slate-800 shadow-2xl z-20 transform transition-transform duration-300 ease-in-out ${alert ? 'translate-x-0' : 'translate-x-full'}`}
    >
      <div className="flex flex-col h-full border-l border-slate-700">
        {/* Header */}
        <div className="flex justify-between items-center p-4 border-b border-slate-700">
          <h2 className="text-xl font-bold text-white">Investigate Alert</h2>
          <button onClick={onClose} className="p-2 rounded-full hover:bg-slate-700">
            <FiX className="h-6 w-6" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-grow p-6 overflow-y-auto space-y-6 scrollbar-thin scrollbar-thumb-slate-600 scrollbar-track-slate-800 hover:scrollbar-thumb-cyan-600">
          {/* User Info */}
          <div className="bg-slate-900 p-4 rounded-lg">
            <h3 className="text-sm font-semibold text-gray-400 mb-2">USER</h3>
            <div className="flex items-center gap-3">
              <FiUser className="w-8 h-8 text-cyan-400" />
              <span className="text-2xl font-mono text-white">{alert.user_id}</span>
            </div>
          </div>

          {/* Score & Severity */}
          <div>
            <h3 className="text-sm font-semibold text-gray-400 mb-2">ANOMALY SCORE</h3>
            <ScoreVisualizer score={alert.anomaly_score} />
            <div className="mt-4">
              <span className={`px-3 py-1 text-sm font-semibold rounded-full bg-${severityColor}-500/20 text-${severityColor}-400`}>
                Threat Level: {levelText}
              </span>
            </div>
          </div>

          {/* Key Factors */}
          <div>
            <h3 className="text-sm font-semibold text-gray-400 mb-2">KEY FACTORS</h3>
            <div className="space-y-2">
              {features.map((f, i) => (
                <div key={i} className="bg-slate-700 p-3 rounded-lg text-sm">
                  <div className="flex justify-between items-center text-gray-400">
                    <span>{f.name}</span>
                    <span className="font-mono text-xs">{(parseFloat(f.importance) * 100).toFixed(1)}% Importance</span>
                  </div>
                  <div className="text-white font-semibold text-base mt-1">{f.value}</div>
                </div>
              ))}
            </div>
          </div>
          
          {/* Analyst Notes */}
          <div>
            <h3 className="text-sm font-semibold text-gray-400 mb-2 flex items-center gap-2"><FiEdit/> ANALYST NOTES</h3>
            <textarea 
              className="w-full bg-slate-700 rounded-lg p-3 text-white focus:outline-none focus:ring-2 focus:ring-cyan-500" 
              rows="4" 
              placeholder="Add investigation notes here..."
            ></textarea>
          </div>
        </div>

        {/* Actions Footer */}
        <div className="p-4 border-t border-slate-700">
          <h3 className="text-sm font-semibold text-gray-400 mb-3">ACTIONS</h3>
          <div className="flex gap-3">
            <select onChange={(e) => onStatusChange(alert.rank, e.target.value)} value={alert.status} className="flex-grow bg-slate-700 border border-slate-600 rounded-lg py-2 px-3 text-white focus:outline-none focus:ring-2 focus:ring-cyan-500">
              <option value="New">New</option>
              <option value="Under Investigation">Under Investigation</option>
              <option value="Contained">Contained</option>
              <option value="Closed">Closed</option>
            </select>
            <button onClick={onClose} className="flex items-center gap-2 bg-cyan-600 hover:bg-cyan-500 text-white font-semibold px-4 py-2 rounded-lg">
              <FiCheckCircle />
              <span>Done</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default InvestigationPanel;