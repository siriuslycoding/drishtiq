// src/components/DataSourceStatus.js
import React from 'react';

const DataSourceStatus = () => {
  const sources = [
    { name: 'System Logs', status: 'Healthy' },
    { name: 'Application Events', status: 'Healthy' },
    { name: 'Endpoint Security', status: 'Warning' },
    { name: 'Server Access Logs', status: 'Healthy' },
    { name: 'Network Devices', status: 'Error' },
  ];

  const getStatusIndicator = (status) => {
    switch (status) {
      case 'Healthy':
        return <div className="h-3 w-3 bg-green-500 rounded-full"></div>;
      case 'Warning':
        return <div className="h-3 w-3 bg-yellow-500 rounded-full"></div>;
      case 'Error':
        return <div className="h-3 w-3 bg-red-500 rounded-full"></div>;
      default:
        return null;
    }
  };

  return (
    <div className="bg-slate-800 p-6 rounded-xl shadow-lg border border-slate-700 h-full">
      <h3 className="font-semibold text-white text-lg mb-4">Data Normalization Pipeline</h3>
       <ul className="space-y-4">
        {sources.map((source, index) => (
          <li key={index} className="flex justify-between items-center">
            <span className="font-medium text-gray-300">{source.name}</span>
            <div className="flex items-center space-x-2">
              {getStatusIndicator(source.status)}
              <span className="text-sm text-gray-400">{source.status}</span>
            </div>
          </li>
        ))}
       </ul>
    </div>
  );
};

export default DataSourceStatus;