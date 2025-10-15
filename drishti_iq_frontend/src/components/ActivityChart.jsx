import React from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Label } from 'recharts';

const ActivityChart = ({ data, loading }) => {
    return (
        <div className="bg-slate-800 p-6 rounded-xl shadow-lg border border-slate-700 h-full">
            <h3 className="font-semibold text-white text-lg mb-4">Anomaly Score Timeline (Sampled Events)</h3>
            {loading ? (
                <div className="h-80 flex items-center justify-center text-gray-500">Loading Chart Data...</div>
            ) : (
                <ResponsiveContainer width="100%" height={320}>
                    <AreaChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 25 }}>
                        <defs>
                            <linearGradient id="colorScore" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#ef4444" stopOpacity={0.8}/>
                                <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                        
                        {/* --- UPDATED X-AXIS --- */}
                        <XAxis 
                            dataKey="user_id"
                            tick={{ fill: '#9ca3af', fontSize: 12 }} 
                        >
                            <Label value="User ID" position="insideBottom" offset={-15} fill="#9ca3af" />
                        </XAxis>
                        
                        {/* --- UPDATED Y-AXIS --- */}
                        <YAxis tick={{ fill: '#9ca3af' }} domain={[0, 1]}>
                            <Label 
                                value="Ensemble Anomaly Score" 
                                angle={-90} 
                                position="insideLeft" 
                                style={{ textAnchor: 'middle', fill: '#9ca3af' }} 
                            />
                        </YAxis>

                        <Tooltip
                            cursor={{ stroke: '#64748b', strokeWidth: 1 }}
                            contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                            labelStyle={{ color: '#d1d5db' }}
                            formatter={(value, name, props) => [`${parseFloat(value).toFixed(4)}`, `User ID: ${props.payload.user_id}`]}
                        />
                        <ReferenceLine y={0.5} label={{ value: 'Threshold', fill: '#facc15', fontSize: 12 }} stroke="#facc15" strokeDasharray="3 3" />
                        <Area type="monotone" dataKey="ensemble_score" stroke="#ef4444" fillOpacity={1} fill="url(#colorScore)" />
                    </AreaChart>
                </ResponsiveContainer>
            )}
        </div>
    );
};

export default ActivityChart;