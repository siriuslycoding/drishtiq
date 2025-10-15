import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { FiCpu, FiTarget, FiTrendingUp, FiZap, FiInfo, FiFileText } from 'react-icons/fi';
import ConfusionMatrix from './ConfusionMatrix'; // Import the new component

const MetricCard = ({ title, value, icon, tooltip }) => (
  <div className="bg-slate-800 p-5 rounded-xl border border-slate-700 group relative">
    <div className="flex items-center gap-4">
      <div className="text-3xl text-cyan-400">{icon}</div>
      <div>
        <p className="text-sm text-gray-400">{title}</p>
        <p className="text-2xl font-bold text-white">{value}</p>
      </div>
    </div>
    {tooltip && <div className="absolute bottom-full mb-2 left-1/2 -translate-x-1/2 w-48 p-2 bg-slate-900 text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity duration-300 z-10">{tooltip}</div>}
  </div>
);

const InterpretationGuide = ({ metrics }) => (
    <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 h-full">
        <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2"><FiInfo /> Interpretation Guide</h3>
        <div className="space-y-4 text-gray-300">
            <p>
                This model is optimized for a <strong className="text-cyan-400">high-recall strategy</strong>, which is critical for security applications.
            </p>
            <p>
                <strong className="text-green-400">High Recall (Sensitivity): {(metrics['1'].recall * 100).toFixed(2)}%</strong><br/>
                This is our key strength. It means the model successfully detects almost all true threats, ensuring minimal risk of letting an attack go unnoticed.
            </p>
            <p>
                <strong className="text-yellow-400">Moderate Precision: {(metrics['1'].precision * 100).toFixed(2)}%</strong><br/>
                This indicates that some alerts may be false positives. This is an intentional trade-off to maximize threat detection. It's better to have an analyst review a false alarm than to miss a real threat.
            </p>
        </div>
    </div>
);

const ModelAnalyticsPage = () => {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        setLoading(true);
        const response = await axios.get('http://localhost:5000/api/metrics');
        setMetrics(response.data);
      } catch (error) {
        console.error("Failed to fetch metrics", error);
      } finally {
        setLoading(false);
      }
    };
    fetchMetrics();
  }, []);

  if (loading) {
    return <div className="p-6 text-center text-gray-500">Loading Model Analytics...</div>;
  }

  if (!metrics) {
    return <div className="p-6 text-center text-red-500">Failed to load model metrics.</div>;
  }

  const { ensemble_metrics: em } = metrics;
  const lastUpdated = new Date(metrics.timestamp).toLocaleString();

  return (
    <div className="p-6 text-white space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold flex items-center gap-3"><FiCpu /> Model Performance Analytics</h1>
        <p className="text-sm text-gray-400">Last Updated: {lastUpdated}</p>
      </div>
      
      {/* Hero Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard title="Overall Accuracy" value={`${(em.accuracy * 100).toFixed(2)}%`} icon={<FiTarget />} tooltip="Percentage of all predictions that were correct." />
        <MetricCard title="ROC AUC Score" value={em.roc_auc_score ? em.roc_auc_score.toFixed(4) : metrics.roc_auc_score.toFixed(4)} icon={<FiTrendingUp />} tooltip="Model's ability to distinguish between classes. Higher is better." />
        <MetricCard title="Total Records" value={metrics.total_records.toLocaleString()} icon={<FiFileText />} tooltip="Total events processed by the model for this report." />
        <MetricCard title="Execution Time" value={`${metrics.execution_time_seconds.toFixed(2)}s`} icon={<FiZap />} tooltip="Total time for the model to train and predict."/>
      </div>

      {/* Main Content: Confusion Matrix and Interpretation */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ConfusionMatrix metrics={em} />
        <InterpretationGuide metrics={em} />
      </div>
    </div>
  );
};

export default ModelAnalyticsPage;