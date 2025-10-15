import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { FiGrid, FiShield, FiUsers, FiCpu, FiEye, FiSettings, FiLogOut, FiChevronsLeft } from 'react-icons/fi';
import DrishtiIQLogo from '../assets/logo2.jpg';

const Sidebar = ({ isCollapsed, onToggle }) => {
  const location = useLocation();

  const navItems = [
    { icon: <FiGrid />, name: 'Dashboard', path: '/' },
    { icon: <FiShield />, name: 'Threat Investigation', path: '/threats' },
    { icon: <FiUsers />, name: 'User Explorer', path: '/users' },
    { icon: <FiCpu />, name: 'Model Analytics', path: '/model-analytics' },
  ];

  return (
    // --- COLLAPSIBLE CONTAINER ---
    <div className={`relative flex flex-col bg-gradient-to-b from-slate-800 to-slate-900 border-r border-slate-700 transition-all duration-300 ease-in-out ${isCollapsed ? 'w-20' : 'w-64'}`}>
      
      {/* --- BRANDING --- */}
      <div className="h-20 flex items-center justify-center gap-3 px-4 mt-5">
        <img src={DrishtiIQLogo} alt="DrishtiIQ Logo" className="h-20 w-20 flex-shrink-0 rounded-2xl" />
      </div>

      {/* --- NAVIGATION --- */}
      <nav className="flex-1 px-3 py-4">
        <ul>
          {navItems.map((item) => {
            const isActive = location.pathname === item.path;
            return (
              <li key={item.name} className="mb-1">
                <Link
                  to={item.path}
                  title={item.name} // Tooltip for collapsed view
                  className={`flex items-center p-3 rounded-lg transition-colors duration-200 relative ${
                    isActive
                      ? 'bg-slate-700/50 text-white'
                      : 'text-gray-400 hover:bg-slate-700 hover:text-white'
                  }`}
                >
                  {/* --- NEW ACTIVE INDICATOR --- */}
                  {isActive && <div className="absolute left-0 top-0 h-full w-1 bg-cyan-400 rounded-r-full"></div>}
                  
                  <span className="text-xl">{item.icon}</span>
                  <span className={`ml-4 overflow-hidden whitespace-nowrap transition-opacity duration-200 ${isCollapsed ? 'opacity-0' : 'opacity-100'}`}>
                    {item.name}
                  </span>
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>

      {/* --- COLLAPSE TOGGLE --- */}
      <div className={`px-3 py-2 border-t border-slate-700`}>
          <button onClick={onToggle} className="w-full flex items-center p-3 rounded-lg text-gray-400 hover:bg-slate-700 hover:text-white">
            <span className={`text-xl transition-transform duration-300 ${isCollapsed ? 'rotate-180' : ''}`}><FiChevronsLeft /></span>
            <span className={`ml-4 overflow-hidden whitespace-nowrap transition-opacity duration-200 ${isCollapsed ? 'opacity-0' : 'opacity-100'}`}>Collapse</span>
          </button>
      </div>

      {/* --- NEW USER PROFILE SECTION --- */}
      <div className="p-3 border-t border-slate-700">
        <div className="flex items-center gap-3 bg-slate-700/30 p-2 rounded-lg">
          <div className="w-10 h-10 rounded-full bg-gradient-to-br from-cyan-400 to-blue-600">
          {/* User Avatar */}
        </div>
          <div className={`overflow-hidden transition-opacity duration-200 ${isCollapsed ? 'opacity-0' : 'opacity-100'}`}>
            <p className="font-semibold text-white text-sm">High Level Schedulers</p>
            <p className="text-xs text-gray-400">Security Analyst</p>
          </div>
          <button className={`ml-auto p-2 rounded-full text-gray-400 hover:bg-slate-600 transition-opacity duration-200 ${isCollapsed ? 'opacity-0' : 'opacity-100'}`}><FiLogOut/></button>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;