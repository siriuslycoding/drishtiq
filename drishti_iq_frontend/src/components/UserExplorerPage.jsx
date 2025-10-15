import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { FiSearch, FiUser, FiTrendingUp, FiAlertOctagon, FiClock, FiDownload, FiFileText } from 'react-icons/fi';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

// --- Reusable Components for this page ---
const RiskyUserListItem = ({ user, onClick, isActive }) => (
  <li 
    onClick={() => onClick(user.user_id)}
    className={`p-3 rounded-lg cursor-pointer transition-colors flex justify-between items-center ${isActive ? 'bg-cyan-600 text-white' : 'hover:bg-slate-700'}`}
  >
    <div className="flex items-center gap-3">
      <FiUser />
      <span className="font-semibold">{user.user_id}</span>
    </div>
    <span className={`font-mono text-sm ${isActive ? 'text-white' : 'text-red-400'}`}>{user.max_score.toFixed(4)}</span>
  </li>
);

const UserDetailPanel = ({ userId }) => {
    const [userData, setUserData] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        if (userId) {
            const fetchDetails = async () => {
                try {
                    setLoading(true);
                    const response = await axios.get(`http://localhost:5000/api/users/${userId}/details`);
                    setUserData(response.data);
                } catch (error) {
                    console.error("Failed to fetch user details", error);
                    setUserData(null);
                } finally {
                    setLoading(false);
                }
            };
            fetchDetails();
        }
    }, [userId]);

    if (loading && userId) {
        return <div className="p-6 text-center text-gray-400">Loading user details...</div>;
    }
    
    if (!userData) {
        return (
            <div className="h-full flex flex-col items-center justify-center text-center p-10 bg-slate-800 rounded-xl border border-dashed border-slate-700">
                <FiUser className="w-16 h-16 text-slate-600 mb-4"/>
                <h3 className="text-xl font-bold text-white">Select a User</h3>
                <p className="text-slate-400">Choose a user from the "Top Risky Users" list to see their detailed profile and activity.</p>
            </div>
        );
    }
    
    const topAlert = userData.alerts[0] || {};

    return (
        <div className="space-y-6">
            {/* Profile Header */}
            <div className="bg-slate-800 p-6 rounded-xl border border-slate-700">
                <div className="flex items-center justify-between">
                    <div>
                        <h2 className="text-3xl font-bold text-white">User Profile: {userData.id}</h2>
                        <p className="text-gray-400">{userData.profile.role} • {userData.profile.department}</p>
                    </div>
                    <div className="text-right">
                        <p className="text-sm text-gray-400">Highest Risk Score</p>
                        <p className="text-4xl font-bold text-red-500">{userData.maxScore.toFixed(4)}</p>
                    </div>
                </div>
            </div>

            {/* Behavioral Baseline */}
            <div>
                <h3 className="text-xl font-semibold mb-4 text-white">Behavioral Baseline vs. Anomaly</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="bg-slate-800 p-4 rounded-lg border border-slate-700">
                        <p className="text-sm text-gray-400 flex items-center gap-2 mb-2"><FiClock/> Login Hour</p>
                        <p className="text-sm">Normal: 9 AM - 10 AM</p>
                        <p className="font-bold text-lg text-yellow-400">Anomaly: {topAlert.feature_2_value || 'N/A'}</p>
                    </div>
                    <div className="bg-slate-800 p-4 rounded-lg border border-slate-700">
                        <p className="text-sm text-gray-400 flex items-center gap-2 mb-2"><FiDownload/> Data Download (MB)</p>
                        <p className="text-sm">Normal: &lt; 10 MB</p>
                        <p className="font-bold text-lg text-yellow-400">Anomaly: {topAlert.feature_1_value || 'N/A'}</p>
                    </div>
                    <div className="bg-slate-800 p-4 rounded-lg border border-slate-700">
                        <p className="text-sm text-gray-400 flex items-center gap-2 mb-2"><FiFileText/> Resource Accesses</p>
                        <p className="text-sm">Normal: 5-10 / day</p>
                        <p className="font-bold text-lg text-yellow-400">Anomaly: {topAlert.access_resources || 'N/A'}</p>
                    </div>
                </div>
            </div>

            {/* Activity Timeline */}
            <div className="bg-slate-800 p-6 rounded-xl border border-slate-700">
              <h3 className="flex items-center gap-2 font-semibold mb-4 text-white"><FiTrendingUp /> Recent Activity & Anomaly Score</h3>
              <ResponsiveContainer width="100%" height={250}>
                <AreaChart data={userData.activity}>
                  <defs><linearGradient id="riskColor" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#ef4444" stopOpacity={0.8}/><stop offset="95%" stopColor="#ef4444" stopOpacity={0}/></linearGradient></defs>
                  
                  {/* --- CHANGE HERE --- */}
                  <Tooltip contentStyle={{ backgroundColor: '#1f2937' }} formatter={(value) => [parseFloat(value).toFixed(4), 'Score']} />
                  {/* --- END CHANGE --- */}

                  <YAxis tick={{ fill: '#9ca3af' }} domain={[0, 1]} />
                  <Area type="monotone" dataKey="ensemble_score" stroke="#ef4444" fill="url(#riskColor)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
        </div>
    );
};

// The main UserExplorerPage component remains the same
const UserExplorerPage = () => {
  const [riskyUsers, setRiskyUsers] = useState([]);
  const [selectedUserId, setSelectedUserId] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    const fetchRiskyUsers = async () => {
      try {
        const response = await axios.get('http://localhost:5000/api/users/riskiest');
        setRiskyUsers(response.data);
        if (response.data.length > 0) {
            setSelectedUserId(response.data[0].user_id);
        }
      } catch (error) {
        console.error("Failed to fetch risky users", error);
      }
    };
    fetchRiskyUsers();
  }, []);
  
  const filteredUsers = riskyUsers.filter(user => user.user_id.toLowerCase().includes(searchTerm.toLowerCase()));

  return (
    <div className="flex h-full p-6 text-white gap-6">
        {/* Left Pane: Risky User List */}
        <div className="w-1/4 bg-slate-800 p-4 rounded-xl border border-slate-700 flex flex-col">
            <h2 className="text-xl font-bold mb-4">Top Risky Users</h2>
            <div className="relative mb-4">
                <FiSearch className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
                <input 
                    type="text" 
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    placeholder="Filter users..."
                    className="bg-slate-700 rounded-lg py-2 pl-10 pr-4 w-full focus:outline-none focus:ring-2 focus:ring-cyan-500"
                />
            </div>
            <ul className="flex-grow overflow-y-auto space-y-2 pr-2 scrollbar">
                {filteredUsers.map(user => (
                    <RiskyUserListItem 
                        key={user.user_id} 
                        user={user} 
                        onClick={setSelectedUserId} 
                        isActive={selectedUserId === user.user_id}
                    />
                ))}
            </ul>
        </div>
        {/* Right Pane: User Details */}
        <div className="w-3/4 flex-grow">
            <UserDetailPanel userId={selectedUserId} />
        </div>
    </div>
  );
};

export default UserExplorerPage;