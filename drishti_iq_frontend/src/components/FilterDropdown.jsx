import React from 'react';
import { FiChevronDown } from 'react-icons/fi';

const FilterDropdown = ({ label, options, value, onChange, icon }) => {
  return (
    <div className="relative">
      <select 
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="appearance-none bg-slate-700 border border-slate-600 rounded-lg py-2 pl-10 pr-8 text-white focus:outline-none focus:ring-2 focus:ring-cyan-500 cursor-pointer"
      >
        {options.map(option => (
          <option key={option.value} value={option.value}>{option.label}</option>
        ))}
      </select>
      <div className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400">{icon}</div>
      <div className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 pointer-events-none">
        <FiChevronDown />
      </div>
    </div>
  );
};

export default FilterDropdown;