// src/components/StatCard.js
import React from 'react';

const StatCard = ({ icon, title, value }) => {
  return (
    <div className="bg-slate-800 p-5 rounded-xl shadow-lg flex items-center justify-between border border-slate-700">
      <div>
        <p className="text-sm text-gray-400 font-medium">{title}</p>
        <p className="text-3xl font-bold text-white">{value}</p>
      </div>
      <div className="text-4xl">
        {icon}
      </div>
    </div>
  );
};

export default StatCard;