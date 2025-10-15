import React, { useEffect, useState } from 'react';
import axios from 'axios';
import StatCard from './StatCard';
import ActivityChart from './ActivityChart';
import AlertsTable from './AlertsTable';
import DataSourceStatus from './DataSourceStatus';
import { FiFileText, FiTarget, FiCheckSquare, FiAward } from 'react-icons/fi';

const Dashboard = () => {
    const [stats, setStats] = useState({});
    const [alerts, setAlerts] = useState([]);
    const [dataSourceHealth, setDataSourceHealth] = useState([]);
    const [timelineData, setTimelineData] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchData = async () => {
            try {
                setLoading(true);
                const [statsRes, alertsRes, dataSourcesRes, timelineRes] = await Promise.all([
                    axios.get('http://localhost:5000/api/dashboard/stats'),
                    axios.get('http://localhost:5000/api/alerts'),
                    axios.get('http://localhost:5000/api/datasources/status'),
                    axios.get('http://localhost:5000/api/analytics/timeline')
                ]);

                setStats(statsRes.data);
                setAlerts(alertsRes.data);
                setDataSourceHealth(dataSourcesRes.data);
                setTimelineData(timelineRes.data);

            } catch (error) {
                console.error('Error fetching data from backend:', error);
            } finally {
                setLoading(false);
            }
        };
        fetchData();
    }, []);

    const formatPercent = (value) => (value * 100).toFixed(2) + '%';
    const formatNumber = (value) => value ? value.toLocaleString() : '0';

    return (
        <div className="space-y-6 p-3">
            {/* Stat Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <StatCard icon={<FiFileText className="text-blue-400"/>} title="Total Records Analyzed" value={loading ? '...' : formatNumber(stats.totalRecords)}/>
                <StatCard icon={<FiTarget className="text-red-400"/>} title="Anomalies in Dataset" value={loading ? '...' : formatNumber(stats.anomaliesFound)}/>
                <StatCard icon={<FiCheckSquare className="text-green-400"/>} title="Model Accuracy" value={loading ? '...' : formatPercent(stats.modelAccuracy)}/>
                <StatCard icon={<FiAward className="text-cyan-400"/>} title="ROC AUC Score" value={loading ? '...' : stats.rocAucScore?.toFixed(4)}/>
            </div>

            {/* Main Content Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="lg:col-span-2">
                    <ActivityChart data={timelineData} loading={loading} />
                </div>
                <div>
                    <DataSourceStatus sources={dataSourceHealth} />
                </div>
            </div>

            <div>
                <AlertsTable alerts={alerts} loading={loading}/>
            </div>
        </div>
    );
};

export default Dashboard;