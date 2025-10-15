import React from 'react';
import { FiSearch, FiBell } from 'react-icons/fi';
import DrishtiIQLogo from '../assets/logo2.jpg'; // Make sure your logo is here

const Header = () => {
  return (
    <header className="bg-slate-800 p-4 flex justify-between items-center border-b border-slate-700">
      <div className="flex items-center">
        <h2 className="text-3xl font-extrabold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">DrishtIQ</h2>
      </div>
      <div className="flex items-center space-x-6">
        <div className="relative">
          <FiSearch className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
          <input
            type="text"
            placeholder="Search threats, users..."
            className="bg-slate-700 border border-slate-600 rounded-full py-2 pl-10 pr-4 w-64 focus:outline-none focus:ring-2 focus:ring-cyan-500"
          />
        </div>
        <button className="relative text-2xl text-gray-400 hover:text-white">
          <FiBell />
          <span className="absolute top-0 right-0 h-2 w-2 bg-red-500 rounded-full"></span>
        </button>
        <div className="w-10 h-10 rounded-full bg-gradient-to-br from-cyan-400 to-blue-600">
          {/* User Avatar */}
        </div>
      </div>
    </header>
  );
};

export default Header;