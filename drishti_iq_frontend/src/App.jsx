import { useState } from 'react'; // Import useState
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

import Sidebar from './components/Sidebar';
import Header from './components/Header';
import Dashboard from './components/Dashboard';
import ThreatInvestigationPage from './components/ThreatInvestigationPage';
import UserExplorerPage from './components/UserExplorerPage';
import ModelAnalyticsPage from './components/ModelAnalyticsPage';

function App() {
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);

  const toggleSidebar = () => {
    setIsSidebarCollapsed(!isSidebarCollapsed);
  };

  return (
    <Router>
      <div className="flex h-screen bg-slate-900 text-gray-300 font-sans">
        <Sidebar isCollapsed={isSidebarCollapsed} onToggle={toggleSidebar} />
        <div className="flex-1 flex flex-col overflow-hidden">
          <Header />
          <main className="flex-1 overflow-x-hidden overflow-y-auto scrollbar">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/threats" element={<ThreatInvestigationPage />} />
              <Route path="/users" element={<UserExplorerPage />} />
              <Route path="/model-analytics" element={<ModelAnalyticsPage />} />
            </Routes>
          </main>
        </div>
      </div>
    </Router>
  );
}

export default App;