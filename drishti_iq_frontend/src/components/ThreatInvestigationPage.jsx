import React, { useState, useEffect, useMemo } from 'react';
import axios from 'axios';
import { FiFilter, FiSearch, FiArrowUp, FiArrowDown, FiCalendar, FiShield } from 'react-icons/fi';
import FilterDropdown from './FilterDropdown';
import InvestigationPanel from './InvestigationPanel';

const ThreatInvestigationPage = () => {
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedAlert, setSelectedAlert] = useState(null);

  // State for interactivity
  const [searchTerm, setSearchTerm] = useState('');
  const [severityFilter, setSeverityFilter] = useState('All');
  const [statusFilter, setStatusFilter] = useState('All');
  const [sortConfig, setSortConfig] = useState({ key: 'anomaly_score', direction: 'desc' });

  useEffect(() => {
    const fetchAlerts = async () => {
      try {
        setLoading(true);
        const response = await axios.get('http://localhost:5000/api/alerts');
        const alertsWithStatus = response.data.map(alert => ({ ...alert, status: 'New' }));
        setAlerts(alertsWithStatus);
      } catch (error) {
        console.error("Failed to fetch alerts", error);
      } finally {
        setLoading(false);
      }
    };
    fetchAlerts();
  }, []);

  const sortedAndFilteredAlerts = useMemo(() => {
    let filtered = [...alerts];
    if (searchTerm) {
      filtered = filtered.filter(a => a.user_id.toLowerCase().includes(searchTerm.toLowerCase()));
    }
    if (severityFilter !== 'All') {
      filtered = filtered.filter(a => a.threat_level.includes(severityFilter));
    }
    if (statusFilter !== 'All') {
      filtered = filtered.filter(a => a.status === statusFilter);
    }
    filtered.sort((a, b) => {
      if (a[sortConfig.key] < b[sortConfig.key]) return sortConfig.direction === 'asc' ? -1 : 1;
      if (a[sortConfig.key] > b[sortConfig.key]) return sortConfig.direction === 'asc' ? 1 : -1;
      return 0;
    });
    return filtered;
  }, [alerts, searchTerm, severityFilter, statusFilter, sortConfig]);

  const handleSort = (key) => {
    let direction = 'asc';
    if (sortConfig.key === key && sortConfig.direction === 'asc') {
      direction = 'desc';
    }
    setSortConfig({ key, direction });
  };

  const handleStatusChange = (rank, newStatus) => {
    setAlerts(prevAlerts => 
      prevAlerts.map(alert => 
        alert.rank === rank ? { ...alert, status: newStatus } : alert
      )
    );
    if (selectedAlert && selectedAlert.rank === rank) {
        setSelectedAlert(prev => ({ ...prev, status: newStatus }));
    }
  };

  const getSortIcon = (key) => {
    if (sortConfig.key !== key) return null;
    return sortConfig.direction === 'asc' ? <FiArrowUp /> : <FiArrowDown />;
  };

  const statusOptions = [
    { value: 'All', label: 'All Statuses' },
    { value: 'New', label: 'New' },
    { value: 'Under Investigation', label: 'Under Investigation' },
    { value: 'Contained', label: 'Contained' },
    { value: 'Closed', label: 'Closed' }
  ];

  const severityOptions = [
    { value: 'All', label: 'All Severities' },
    { value: 'HIGH', label: 'High' },
    { value: 'MEDIUM', label: 'Medium' },
    { value: 'LOW', label: 'Low' }
  ];

  const getStatusBadge = (status) => {
    switch (status) {
        case 'New': return 'bg-blue-500/20 text-blue-400';
        case 'Under Investigation': return 'bg-yellow-500/20 text-yellow-400';
        case 'Contained': return 'bg-purple-500/20 text-purple-400';
        case 'Closed': return 'bg-green-500/20 text-green-400';
        default: return 'bg-gray-500/20 text-gray-400';
    }
  };

  return (
    <>
      <div className="p-6 text-white">
        <h1 className="text-3xl font-bold mb-6">Threat Investigation Hub</h1>

        <div className="flex flex-col md:flex-row justify-between items-center mb-6 gap-4 bg-slate-800 p-4 rounded-xl border border-slate-700">
          <div className="flex items-center gap-4">
            <FilterDropdown icon={<FiShield/>} options={severityOptions} value={severityFilter} onChange={setSeverityFilter} />
            <FilterDropdown icon={<FiFilter/>} options={statusOptions} value={statusFilter} onChange={setStatusFilter} />
          </div>
          <div className="relative">
            <FiSearch className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
            <input type="text" placeholder="Search by User ID..." value={searchTerm} onChange={(e) => setSearchTerm(e.target.value)} className="bg-slate-700 border border-slate-600 rounded-lg py-2 pl-10 pr-4 w-72 focus:outline-none focus:ring-2 focus:ring-cyan-500" />
          </div>
        </div>

        <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="bg-slate-900/50">
                <tr className="text-gray-400">
                  <th className="p-4 font-semibold cursor-pointer" onClick={() => handleSort('user_id')}>
                    <div className="flex items-center gap-2">User ID {getSortIcon('user_id')}</div>
                  </th>
                  <th className="p-4 font-semibold cursor-pointer" onClick={() => handleSort('anomaly_score')}>
                    <div className="flex items-center gap-2">Anomaly Score {getSortIcon('anomaly_score')}</div>
                  </th>
                  <th className="p-4 font-semibold">Threat Level</th>
                  <th className="p-4 font-semibold">Status</th>
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  <tr><td colSpan="4" className="text-center p-8 text-gray-500">Loading alerts...</td></tr>
                ) : (
                  sortedAndFilteredAlerts.map(alert => (
                    <tr key={alert.rank} onClick={() => setSelectedAlert(alert)} className="border-t border-slate-700 hover:bg-slate-700/50 transition-colors cursor-pointer">
                      <td className="p-4 font-mono text-cyan-400">{alert.user_id}</td>
                      <td className="p-4 font-mono text-white">{parseFloat(alert.anomaly_score).toFixed(4)}</td>
                      <td className="p-4 text-yellow-400">{alert.threat_level.replace(/[^a-zA-Z]/g, '')}</td>
                      <td className="p-4"><span className={`px-2 py-1 rounded-full text-xs font-semibold ${getStatusBadge(alert.status)}`}>{alert.status}</span></td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
      <InvestigationPanel alert={selectedAlert} onClose={() => setSelectedAlert(null)} onStatusChange={handleStatusChange}/>
    </>
  );
};

export default ThreatInvestigationPage;