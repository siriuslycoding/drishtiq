const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const csv = require('csv-parser');

const app = express();
const PORT = 5000;

// Middleware
app.use(cors());
app.use(express.json());

// --- Helper Functions to Read Data ---

const readCsvFile = (fileName) => {
    return new Promise((resolve, reject) => {
        const results = [];
        const filePath = path.join(__dirname, 'data', fileName);
        fs.createReadStream(filePath)
            .pipe(csv())
            .on('data', (data) => results.push(data))
            .on('end', () => resolve(results))
            .on('error', (error) => reject(error));
    });
};

const readJsonFile = (fileName) => {
    return new Promise((resolve, reject) => {
        const filePath = path.join(__dirname, 'data', fileName);
        fs.readFile(filePath, 'utf8', (err, data) => {
            if (err) return reject(err);
            try {
                resolve(JSON.parse(data));
            } catch (parseError) {
                reject(parseError);
            }
        });
    });
};

// --- API ENDPOINTS ---

// Endpoint for the main alerts table, powered by fast_explanations.csv
app.get('/api/alerts', async (req, res) => {
    try {
        const explanations = await readCsvFile('fast_explanations.csv');
        res.json(explanations);
    } catch (error) {
        console.error('Failed to read fast_explanations.csv:', error);
        res.status(500).json({ message: 'Error processing alert explanation data' });
    }
});

// Endpoint for dashboard KPIs, powered by metrics.json
app.get('/api/dashboard/stats', async (req, res) => {
    try {
        const metrics = await readJsonFile('metrics.json');
        
        // Extract the most relevant stats for the cards
        const stats = {
            totalRecords: metrics.total_records,
            anomaliesFound: metrics.anomalies_in_dataset,
            modelAccuracy: metrics.ensemble_metrics.accuracy,
            rocAucScore: metrics.roc_auc_score,
        };
        res.json(stats);
    } catch (error) {
        console.error('Failed to process metrics.json:', error);
        res.status(500).json({ message: 'Error processing stats data' });
    }
});

// Endpoint for the activity timeline chart, powered by predictions.csv
app.get('/api/analytics/timeline', async (req, res) => {
    try {
        const predictions = await readCsvFile('predictions.csv');
        // To avoid crashing the browser, we'll only send a sample (e.g., first 200 records)
        const timelineSample = predictions.slice(0, 200); 
        timelineSample.sort((a, b) => parseInt(a.user_id, 10) - parseInt(b.user_id, 10));
        res.json(timelineSample);
    } catch (error) {
        console.error('Failed to process predictions.csv:', error);
        res.status(500).json({ message: 'Error processing timeline data' });
    }
});

// Static endpoint for data source status
app.get('/api/datasources/status', (req, res) => {
    res.json([
        { name: 'Endpoint Logs', status: 'Healthy' },
        { name: 'Authentication Logs', status: 'Healthy' },
        { name: 'VPN Logs', status: 'Warning' },
        { name: 'Database Access', status: 'Healthy' },
        { name: 'Cloud Provider API', status: 'Error' },
    ]);
});

app.get('/api/metrics', async (req, res) => {
    try {
        const metrics = await readJsonFile('metrics.json');
        res.json(metrics);
    } catch (error) {
        console.error("Error in /api/metrics:", error.message);
        res.status(500).json({ message: 'Error processing metrics data. Check server logs.' });
    }
});

// Endpoint to get a ranked list of the riskiest users
app.get('/api/users/riskiest', async (req, res) => {
    try {
        const explanations = await readCsvFile('fast_explanations.csv');
        
        // Group by user_id and find the highest anomaly score for each user
        const userScores = explanations.reduce((acc, alert) => {
            const score = parseFloat(alert.anomaly_score);
            if (!acc[alert.user_id] || score > acc[alert.user_id]) {
                acc[alert.user_id] = score;
            }
            return acc;
        }, {});

        // Convert to an array, sort by score descending, and take the top 10
        const sortedUsers = Object.entries(userScores)
            .map(([user_id, max_score]) => ({ user_id, max_score }))
            .sort((a, b) => b.max_score - a.max_score)
            .slice(0, 10);

        res.json(sortedUsers);
    } catch (error) {
        console.error("Error in /api/users/riskiest:", error.message);
        res.status(500).json({ message: 'Error processing riskiest users.' });
    }
});

// Endpoint to get ALL details for a specific user
app.get('/api/users/:id/details', async (req, res) => {
    const { id } = req.params;
    try {
        const predictions = await readCsvFile('predictions.csv');
        const explanations = await readCsvFile('fast_explanations.csv');

        const userActivity = predictions.filter(p => p.user_id === id).slice(-100); // Last 100 events
        const userAlerts = explanations.filter(e => e.user_id === id);

        if (userActivity.length === 0 && userAlerts.length === 0) {
            return res.status(404).json({ message: 'User not found' });
        }
        
        // Find the user's highest score for the profile card
        const maxScore = userAlerts.reduce((max, alert) => Math.max(max, parseFloat(alert.anomaly_score)), 0);

        res.json({
            id,
            maxScore,
            activity: userActivity,
            alerts: userAlerts,
            // Mock profile data - in a real system this would come from an HR database
            profile: {
                department: 'Engineering',
                role: 'Senior Developer',
                manager: 'Jane Smith',
                location: 'Pune, IND'
            }
        });

    } catch (error) {
        console.error(`Error fetching details for user ${id}:`, error.message);
        res.status(500).json({ message: 'Error processing user details.' });
    }
});

// Start the server
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});