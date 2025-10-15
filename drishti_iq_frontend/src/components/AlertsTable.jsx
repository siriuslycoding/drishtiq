import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import ScoreVisualizer from './ScoreVisualizer';
import { FiChevronDown, FiChevronUp, FiMoreVertical, FiUserCheck, FiArchive, FiShield } from 'react-icons/fi';

const ExplanationDetails = ({ alert }) => {
  const features = [
    { name: alert.top_feature_1, value: alert.feature_1_value, importance: alert.feature_1_importance },
    { name: alert.top_feature_2, value: alert.feature_2_value, importance: alert.feature_2_importance },
    { name: alert.top_feature_3, value: alert.feature_3_value, importance: alert.feature_3_importance },
  ].filter(f => f.name); // Filter out empty features

  return (
    <div className="bg-slate-900 p-4">
      <h4 className="font-semibold text-gray-300 mb-3">Key Factors Contributing to Anomaly Score:</h4>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
        {features.map((f, i) => (
          <div key={i} className="bg-slate-800 p-3 rounded-lg">
            <div className="flex justify-between items-center text-gray-400 mb-1">
              <span>{f.name}</span>
              <span className="font-mono text-xs">{(parseFloat(f.importance) * 100).toFixed(1)}%</span>
            </div>
            <div className="text-white font-semibold text-lg">{f.value}</div>
          </div>
        ))}
      </div>
    </div>
  );
};

const AlertRow = ({ alert, isExpanded, onToggle }) => {
  const [actionsOpen, setActionsOpen] = useState(false);
  const levelText = alert.threat_level.replace(/[^a-zA-Z]/g, '').toUpperCase();
  const severityColor = levelText === 'HIGH' ? 'red' : levelText === 'MEDIUM' ? 'yellow' : 'blue';

  return (
    <>
      <tr className="hover:bg-slate-700/50 cursor-pointer" onClick={() => onToggle(alert.rank)}>
        <td className="p-4 whitespace-nowrap">
          <Link 
            to={`/users?id=${alert.user_id}`} 
            onClick={(e) => e.stopPropagation()}
            className="font-mono text-cyan-400 hover:underline"
          >
            {alert.user_id}
          </Link>
        </td>
        <td className="p-4 w-64"><ScoreVisualizer score={alert.anomaly_score} /></td>
        <td className="p-4 whitespace-nowrap">
          <span className={`px-3 py-1 text-xs font-semibold rounded-full bg-${severityColor}-500/20 text-${severityColor}-400`}>
            {levelText}
          </span>
        </td>
        <td className="p-4 text-gray-300">{alert.threat_description}</td>
        <td className="p-4 whitespace-nowrap text-gray-400">New</td>
        <td className="p-4 whitespace-nowrap text-center relative">
          <button onClick={(e) => { e.stopPropagation(); setActionsOpen(!actionsOpen); }} className="p-2 rounded-full hover:bg-slate-600">
            <FiMoreVertical />
          </button>
          {actionsOpen && (
            <div className="absolute right-0 mt-2 w-48 bg-slate-800 border border-slate-700 rounded-lg shadow-xl z-10">
              <a href="#" className="flex items-center gap-3 px-4 py-2 text-sm text-gray-300 hover:bg-slate-700"><FiUserCheck /> Investigate User</a>
              <a href="#" className="flex items-center gap-3 px-4 py-2 text-sm text-gray-300 hover:bg-slate-700"><FiShield /> Add to Case</a>
              <a href="#" className="flex items-center gap-3 px-4 py-2 text-sm text-gray-300 hover:bg-slate-700"><FiArchive /> Dismiss Alert</a>
            </div>
          )}
        </td>
        <td className="p-4 whitespace-nowrap text-center">
          {isExpanded ? <FiChevronUp /> : <FiChevronDown />}
        </td>
      </tr>
      {isExpanded && (
        <tr>
          <td colSpan="7" className="p-0">
            <ExplanationDetails alert={alert} />
          </td>
        </tr>
      )}
    </>
  );
};

const AlertsTable = ({ alerts, loading }) => {
  const [expandedRow, setExpandedRow] = useState(null);

  const handleToggleRow = (rank) => {
    setExpandedRow(expandedRow === rank ? null : rank);
  };

  return (
    <div className="bg-slate-800 rounded-xl shadow-lg border border-slate-700 overflow-hidden">
      <h3 className="font-semibold text-white text-lg p-4 border-b border-slate-700">High-Risk Behavior Explanations</h3>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead className="bg-slate-900/50">
            <tr className="text-gray-400">
              <th className="p-4 font-semibold">User ID</th>
              <th className="p-4 font-semibold">Anomaly Score</th>
              <th className="p-4 font-semibold">Threat Level</th>
              <th className="p-4 font-semibold">Description</th>
              <th className="p-4 font-semibold">Status</th>
              <th className="p-4 font-semibold text-center">Actions</th>
              <th className="p-4 font-semibold"></th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
                <tr><td colSpan="7" className="p-8 text-center text-gray-500">Loading alerts...</td></tr>
            ) : alerts.length > 0 ? (
              alerts.map((alert) => (
                <AlertRow 
                  key={alert.rank} 
                  alert={alert}
                  isExpanded={expandedRow === alert.rank}
                  onToggle={handleToggleRow}
                />
              ))
            ) : (
              <tr><td colSpan="7" className="p-8 text-center text-gray-500">No high-risk alerts to display.</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default AlertsTable;