import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
import os
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import time
from datetime import datetime

warnings.filterwarnings('ignore')
sns.set_style("darkgrid")

# ============================================================================
# COLOR & STYLING
# ============================================================================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*90}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}  {text}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*90}{Colors.ENDC}\n")

def print_section(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'─'*90}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}  ► {text}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'─'*90}{Colors.ENDC}\n")

def print_success(text):
    print(f"{Colors.GREEN}  ✓ {text}{Colors.ENDC}")

def print_info(text):
    print(f"{Colors.CYAN}  ℹ {text}{Colors.ENDC}")

def print_warning(text):
    print(f"{Colors.YELLOW}  ⚠ {text}{Colors.ENDC}")

def print_error(text):
    print(f"{Colors.RED}  ✗ {text}{Colors.ENDC}")

def print_data(label, value, color=Colors.BOLD):
    print(f"{color}  {label}: {Colors.ENDC}{Colors.BOLD}{value}{Colors.ENDC}")

# ============================================================================
# SETUP: CREATE FOLDER STRUCTURE
# ============================================================================

def setup_directories():
    """Create dataset folder structure"""
    base_path = Path(__file__).parent / "../dataset"
    base_path = base_path.resolve()
    input_path = base_path / "input"
    output_path = base_path / "output"
    viz_path = output_path / "visualizations"
    
    input_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)
    viz_path.mkdir(parents=True, exist_ok=True)
    
    return base_path, input_path, output_path, viz_path

BASE_PATH, INPUT_PATH, OUTPUT_PATH, VIZ_PATH = setup_directories()

# ============================================================================
# PART 1: MOCK DATA GENERATION
# ============================================================================

class MockDataGenerator:
    """Generate realistic mock user behavior data"""
    
    def __init__(self, n_users=500, n_days=90, n_records_per_day=50):
        self.n_users = n_users
        self.n_days = n_days
        self.n_records_per_day = n_records_per_day
        self.total_records = n_users * n_days * n_records_per_day
        
    def generate_baseline_behavior(self):
        """Create normal user behavior patterns"""
        np.random.seed(42)
        
        data = []
        timestamps = pd.date_range('2024-01-01', periods=self.n_days, freq='D')
        
        for user_id in range(self.n_users):
            normal_login_time = np.random.randint(7, 11)
            normal_logout_time = np.random.randint(17, 19)
            normal_failed_logins_per_day = np.random.choice([0, 0, 1], p=[0.7, 0.25, 0.05])
            normal_file_downloads = np.random.randint(2, 10)
            normal_data_uploaded = np.random.randint(100, 500)
            normal_emails_sent = np.random.randint(10, 50)
            
            for day_idx, date in enumerate(timestamps):
                for _ in range(self.n_records_per_day):
                    hour = np.random.normal(normal_login_time, 2)
                    
                    login_hour = np.clip(int(hour), 0, 23)
                    login_success = np.random.choice([0, 1], p=[normal_failed_logins_per_day/10, 1-normal_failed_logins_per_day/10])
                    file_downloaded = np.random.poisson(normal_file_downloads / self.n_records_per_day)
                    file_uploaded = np.random.poisson(5)
                    data_downloaded_mb = np.random.exponential(normal_data_uploaded / self.n_records_per_day)
                    data_uploaded_mb = np.random.exponential(50)
                    emails_sent = np.random.poisson(normal_emails_sent / self.n_records_per_day)
                    access_resources = np.random.randint(1, 20)
                    
                    external_ip_access = 1 if np.random.random() < 0.05 else 0
                    privilege_escalation = 1 if np.random.random() < 0.01 else 0
                    vpn_usage = 1 if login_hour < 8 or login_hour > 18 else 0
                    copy_removable_media = 1 if np.random.random() < 0.02 else 0
                    is_weekend = (date.weekday() >= 5)
                    
                    data.append({
                        'timestamp': date + pd.Timedelta(hours=login_hour, minutes=np.random.randint(0, 59)),
                        'user_id': user_id,
                        'login_hour': login_hour,
                        'login_success': login_success,
                        'failed_login_attempts': np.random.poisson(normal_failed_logins_per_day) if login_success == 0 else 0,
                        'file_downloaded': file_downloaded,
                        'file_uploaded': file_uploaded,
                        'data_downloaded_mb': max(0, data_downloaded_mb),
                        'data_uploaded_mb': max(0, data_uploaded_mb),
                        'emails_sent': emails_sent,
                        'access_resources': access_resources,
                        'external_ip_access': external_ip_access,
                        'privilege_escalation': privilege_escalation,
                        'vpn_usage': vpn_usage,
                        'copy_removable_media': copy_removable_media,
                        'is_weekend': int(is_weekend),
                        'day_of_week': date.weekday(),
                    })
        
        return pd.DataFrame(data)
    
    def inject_anomalies(self, df, anomaly_percentage=2):
        """Inject insider threat patterns into data"""
        n_anomalies = int(len(df) * (anomaly_percentage / 100))
        anomaly_indices = np.random.choice(df.index, n_anomalies, replace=False)
        
        threat_types = ['data_exfiltration', 'privilege_abuse', 'unusual_access', 'credential_misuse']
        
        for idx in anomaly_indices:
            threat = np.random.choice(threat_types)
            
            if threat == 'data_exfiltration':
                df.loc[idx, 'data_downloaded_mb'] *= np.random.uniform(5, 20)
                df.loc[idx, 'file_downloaded'] *= np.random.uniform(3, 8)
                df.loc[idx, 'login_hour'] = np.random.choice([2, 3, 4, 22, 23])
                df.loc[idx, 'is_weekend'] = 1
                
            elif threat == 'privilege_abuse':
                df.loc[idx, 'privilege_escalation'] = 1
                df.loc[idx, 'access_resources'] *= 3
                
            elif threat == 'unusual_access':
                df.loc[idx, 'external_ip_access'] = 1
                df.loc[idx, 'vpn_usage'] = 0
                df.loc[idx, 'login_hour'] = np.random.randint(0, 6)
                
            elif threat == 'credential_misuse':
                df.loc[idx, 'failed_login_attempts'] = np.random.randint(5, 15)
                df.loc[idx, 'login_success'] = 1
                df.loc[idx, 'emails_sent'] *= 2
            
            df.loc[idx, 'anomaly_label'] = 1
        
        df['anomaly_label'] = df['anomaly_label'].fillna(0).astype(int)
        return df
    
    def save_raw_data(self, df):
        """Save raw generated data to input folder"""
        filepath = INPUT_PATH / "raw_data.csv"
        df.to_csv(filepath, index=False)
        print_success(f"Saved raw data: {filepath} ({len(df):,} records)")

# ============================================================================
# PART 2: FEATURE ENGINEERING
# ============================================================================

class FeatureEngineer:
    """Create behavioral features for anomaly detection"""
    
    @staticmethod
    def engineer_features(df):
        """Create aggregate features from raw logs"""
        
        df['hour_of_day'] = df['login_hour']
        df['is_business_hours'] = ((df['login_hour'] >= 9) & (df['login_hour'] <= 17)).astype(int)
        df['is_night_time'] = ((df['login_hour'] < 6) | (df['login_hour'] > 20)).astype(int)
        
        user_stats = df.groupby('user_id').agg({
            'data_downloaded_mb': ['mean', 'std', 'max'],
            'data_uploaded_mb': ['mean', 'std', 'max'],
            'file_downloaded': ['mean', 'max'],
            'emails_sent': ['mean', 'max'],
            'failed_login_attempts': ['mean', 'sum'],
            'privilege_escalation': 'sum',
            'external_ip_access': 'sum',
            'copy_removable_media': 'sum',
            'access_resources': ['mean', 'max'],
        }).fillna(0)
        
        user_stats.columns = ['_'.join(col).strip() for col in user_stats.columns.values]
        user_stats = user_stats.reset_index()
        
        df = df.merge(user_stats, on='user_id', how='left')
        
        df['download_deviation'] = (df['data_downloaded_mb'] - df['data_downloaded_mb_mean']) / (df['data_downloaded_mb_std'] + 1)
        df['upload_deviation'] = (df['data_uploaded_mb'] - df['data_uploaded_mb_mean']) / (df['data_uploaded_mb_std'] + 1)
        df['file_activity_deviation'] = (df['file_downloaded'] - df['file_downloaded_mean']) / 2
        
        df['risk_score'] = (
            (df['external_ip_access'] * 3) +
            (df['privilege_escalation'] * 5) +
            (df['copy_removable_media'] * 4) +
            (df['failed_login_attempts'] * 2) +
            (df['is_night_time'] * 1) +
            (df['vpn_usage'] * 0.5)
        )
        
        return df

# ============================================================================
# PART 3: MODEL TRAINING
# ============================================================================

class UEBAModel:
    """UEBA anomaly detection model"""
    
    def __init__(self):
        self.isolation_forest = IsolationForest(
            contamination=0.02,
            random_state=42,
            n_estimators=100,
            max_samples='auto'
        )
        self.rf_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def train(self, X_train, y_train=None):
        """Train model - unsupervised and supervised approaches"""
        
        self.feature_columns = X_train.columns.tolist()
        X_scaled = self.scaler.fit_transform(X_train)
        
        self.isolation_forest.fit(X_scaled)
        
        if y_train is not None:
            self.rf_classifier.fit(X_scaled, y_train)
            return "Trained both Isolation Forest and Random Forest"
        return "Trained Isolation Forest (unsupervised)"
    
    def predict(self, X_test):
        """Generate predictions and anomaly scores"""
        X_scaled = self.scaler.transform(X_test)
        
        if_predictions = self.isolation_forest.predict(X_scaled)
        if_scores = -self.isolation_forest.score_samples(X_scaled)
        
        rf_scores = None
        if hasattr(self.rf_classifier, 'classes_'):
            rf_proba = self.rf_classifier.predict_proba(X_scaled)
            rf_scores = rf_proba[:, 1]
        
        ensemble_score = if_scores.copy()
        if rf_scores is not None:
            ensemble_score = (if_scores + rf_scores) / 2
        
        return {
            'if_prediction': if_predictions,
            'if_score': if_scores,
            'rf_score': rf_scores,
            'ensemble_score': ensemble_score,
            'is_anomaly': (ensemble_score > np.percentile(ensemble_score, 95)).astype(int),
            'X_scaled': X_scaled
        }

# ============================================================================
# PART 4: FAST FEATURE IMPORTANCE BASED EXPLANATIONS
# ============================================================================
class FastExplainer:
    """Fast explanation using Random Forest feature importance + permutation."""
    
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        self.importances = model.rf_classifier.feature_importances_

    def create_explanation_df(self, predictions, user_ids, X_test):
        """Create detailed explanation dataframe - FAST METHOD."""
        explanations = []

        # Get top 25 anomalies
        top_indices = np.argsort(predictions)[-25:][::-1]

        for rank, idx in enumerate(top_indices, 1):
            # Get feature values for this record
            feature_vals = X_test.iloc[idx]

            # Calculate contribution: importance * feature_value
            contributions = np.abs(self.importances) * np.abs(feature_vals)
            top_3_idx = np.argsort(contributions)[-3:][::-1]

            top_features = [
                (self.feature_names[i], self.importances[i], feature_vals.iloc[i])
                for i in top_3_idx
            ]

            threat_desc = self.generate_threat_description(top_features, feature_vals)
            threat_level = self.calculate_threat_level(predictions[idx])

            # Use index-style access on user_ids (works for list, Index, Series)
            try:
                uid = user_ids[idx]
            except Exception:
                uid = list(user_ids)[idx]  # fallback for unexpected data types

            explanation = {
                'rank': rank,
                'user_id': uid,
                'anomaly_score': f"{predictions[idx]:.4f}",
                'top_feature_1': top_features[0][0],
                'feature_1_importance': f"{top_features[0][1]:.4f}",
                'feature_1_value': f"{top_features[0][2]:.2f}",
                'top_feature_2': top_features[1][0],
                'feature_2_importance': f"{top_features[1][1]:.4f}",
                'feature_2_value': f"{top_features[1][2]:.2f}",
                'top_feature_3': top_features[2][0],
                'feature_3_importance': f"{top_features[2][1]:.4f}",
                'feature_3_value': f"{top_features[2][2]:.2f}",
                'threat_description': threat_desc,
                'threat_level': threat_level,
                'action_required': self.get_action(threat_level)
            }

            explanations.append(explanation)

        return pd.DataFrame(explanations)

    def generate_threat_description(self, top_features, feature_vals):
        """Generate human-readable threat description."""
        threats = []

        for feat_name, importance, value in top_features:
            if 'privilege_escalation' in feat_name and value > 0.5:
                threats.append("🔓 Privilege Escalation")
            elif 'data_downloaded' in feat_name and value > 500:
                threats.append("📥 Massive Data Download")
            elif 'external_ip' in feat_name and value > 0.5:
                threats.append("🌐 External IP Access")
            elif 'night_time' in feat_name and value > 0.5:
                threats.append("🌙 Off-Hours Access")
            elif 'copy_removable' in feat_name and value > 0.5:
                threats.append("💾 USB Copy Activity")
            elif 'failed_login' in feat_name and value > 3:
                threats.append("🔐 Failed Login Attempts")
            elif 'emails_sent' in feat_name and value > 100:
                threats.append("📧 Bulk Email Activity")

        return " | ".join(threats[:2]) if threats else "Anomalous Behavior"

    def calculate_threat_level(self, score):
        """Calculate threat level based on anomaly score."""
        if score > 0.85:
            return "🔴 CRITICAL"
        elif score > 0.70:
            return "🟠 HIGH"
        elif score > 0.50:
            return "🟡 MEDIUM"
        else:
            return "🟢 LOW"

    def get_action(self, threat_level):
        """Get recommended action based on threat level."""
        if "CRITICAL" in threat_level:
            return "Immediate investigation + suspend account"
        elif "HIGH" in threat_level:
            return "Urgent review required"
        elif "MEDIUM" in threat_level:
            return "Monitor closely"
        else:
            return "Standard monitoring"

# ============================================================================
# PART 5: COLORFUL VISUALIZATIONS
# ============================================================================

def create_visualizations(results, y_test, feature_names, X_test, model):
    """Create stunning colorful visualizations"""
    
    print_section("Creating Colorful Visualizations")
    
    fpr, tpr, _ = roc_curve(y_test, results['ensemble_score'])
    roc_auc = roc_auc_score(y_test, results['ensemble_score'])
    
    # Set color palette
    colors = {
        'normal': '#51CF66',
        'anomaly': '#FF6B6B',
        'accent1': '#4ECDC4',
        'accent2': '#FFE66D',
        'accent3': '#95E1D3'
    }
    
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    fig.patch.set_facecolor('#F8F9FA')
    
    # 1. ROC Curve (Large)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor('#FFFFFF')
    ax1.plot(fpr, tpr, color=colors['anomaly'], lw=4, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='#CCCCCC', lw=2, linestyle='--', label='Random Classifier')
    ax1.fill_between(fpr, tpr, alpha=0.2, color=colors['anomaly'])
    ax1.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax1.set_title('🎯 ROC Curve - Model Discrimination', fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc='lower right', fontsize=11, framealpha=0.95)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 2. Model Score Summary
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    summary_text = f"""
    {Colors.BOLD}MODEL METRICS{Colors.ENDC}
    
    ROC-AUC: {roc_auc:.4f}
    
    Threshold: {np.percentile(results['ensemble_score'], 95):.4f}
    
    Anomalies Detected: {results['is_anomaly'].sum()}
    
    Detection Rate: {results['is_anomaly'].sum() / len(results['is_anomaly']) * 100:.2f}%
    """
    ax2.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor=colors['accent1'], alpha=0.3, pad=1))
    
    # 3. Confusion Matrix (Heatmap)
    ax3 = fig.add_subplot(gs[1, 0])
    cm = confusion_matrix(y_test, results['is_anomaly'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn_r', ax=ax3, cbar=True,
                xticklabels=['🟢 Normal', '🔴 Anomaly'], 
                yticklabels=['🟢 Normal', '🔴 Anomaly'],
                annot_kws={'size': 12, 'fontweight': 'bold'}, 
                cbar_kws={'label': 'Count'})
    ax3.set_ylabel('True Label', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    ax3.set_title('🎲 Confusion Matrix', fontsize=13, fontweight='bold', pad=10)
    
    # 4. Anomaly Score Distribution
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor('#FFFFFF')
    ax4.hist(results['ensemble_score'][y_test == 0], bins=60, alpha=0.7, 
            label='Normal', color=colors['normal'], edgecolor='black', linewidth=0.5)
    ax4.hist(results['ensemble_score'][y_test == 1], bins=60, alpha=0.7, 
            label='Anomaly', color=colors['anomaly'], edgecolor='black', linewidth=0.5)
    threshold = np.percentile(results['ensemble_score'], 95)
    ax4.axvline(threshold, color='#FF1744', linestyle='--', linewidth=3, label=f'Threshold ({threshold:.2f})')
    ax4.set_xlabel('Ensemble Anomaly Score', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax4.set_title('📊 Score Distribution', fontsize=13, fontweight='bold', pad=10)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    # 5. Threat Level Distribution (Pie)
    ax5 = fig.add_subplot(gs[1, 2])
    threat_counts = pd.Series(results['is_anomaly']).value_counts()
    colors_pie = [colors['normal'], colors['anomaly']]
    wedges, texts, autotexts = ax5.pie(threat_counts.values, labels=['🟢 Normal', '🔴 Anomaly'],
                                         autopct='%1.1f%%', colors=colors_pie, startangle=90,
                                         textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax5.set_title('🥧 Data Distribution', fontsize=13, fontweight='bold', pad=10)
    
    # 6. Top 15 Feature Importance
    ax6 = fig.add_subplot(gs[2, :])
    importances = model.rf_classifier.feature_importances_
    top_features = np.argsort(importances)[-15:]
    top_importances = importances[top_features]
    top_names = [feature_names[i] for i in top_features]
    
    # Color based on importance
    feat_colors = [colors['anomaly'] if imp > 0.08 else colors['accent1'] for imp in top_importances]
    
    bars = ax6.barh(top_names, top_importances, color=feat_colors, edgecolor='black', linewidth=1.5)
    ax6.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax6.set_title('⭐ Top 15 Feature Importance (Random Forest)', fontsize=14, fontweight='bold', pad=15)
    ax6.grid(True, alpha=0.3, axis='x')
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, top_importances)):
        ax6.text(val + 0.002, bar.get_y() + bar.get_height()/2, f'{val:.4f}',
                va='center', fontsize=9, fontweight='bold')
    
    plt.suptitle('🛡️  UEBA MODEL PERFORMANCE DASHBOARD', fontsize=18, fontweight='bold', y=0.995)
    
    plt.savefig(VIZ_PATH / 'model_performance_dashboard.png', dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
    print_success("Saved: model_performance_dashboard.png")
    plt.close()
    
    # 2nd Visualization: Top Anomalies
    print_info("Creating top anomalies visualization...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('#F8F9FA')
    ax.set_facecolor('#FFFFFF')
    
    top_20_idx = np.argsort(results['ensemble_score'])[-20:][::-1]
    top_20_scores = results['ensemble_score'][top_20_idx]
    top_20_users = X_test.index[top_20_idx]
    
    colors_bars = [colors['anomaly'] if score > 0.85 else colors['accent2'] if score > 0.70 else colors['accent1'] 
                   for score in top_20_scores]
    
    bars = ax.barh(range(len(top_20_scores)), top_20_scores, color=colors_bars, edgecolor='black', linewidth=1.5)
    ax.set_yticks(range(len(top_20_scores)))
    ax.set_yticklabels([f'User {uid}' for uid in top_20_users], fontsize=10, fontweight='bold')
    ax.set_xlabel('Anomaly Score', fontsize=12, fontweight='bold')
    ax.set_title('🚨 Top 20 Most Suspicious Users', fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='x')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, 1.0)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, top_20_scores)):
        threat = "🔴 CRITICAL" if val > 0.85 else "🟠 HIGH" if val > 0.70 else "🟡 MEDIUM"
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.3f} {threat}',
               va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(VIZ_PATH / 'top_anomalies.png', dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
    print_success("Saved: top_anomalies.png")
    plt.close()

# ============================================================================
# PART 6: CHECK IF DATA EXISTS
# ============================================================================

def data_exists():
    """Check if processed data already exists"""
    required_files = [
        INPUT_PATH / "raw_data.csv",
        INPUT_PATH / "X_train.csv",
        INPUT_PATH / "X_test.csv",
        INPUT_PATH / "y_train.csv",
        INPUT_PATH / "y_test.csv"
    ]
    return all(f.exists() for f in required_files)

def load_existing_data():
    """Load existing processed data"""
    print_info("Loading existing data...")
    df = pd.read_csv(INPUT_PATH / "raw_data.csv")
    X_train = pd.read_csv(INPUT_PATH / "X_train.csv")
    X_test = pd.read_csv(INPUT_PATH / "X_test.csv")
    y_train = pd.read_csv(INPUT_PATH / "y_train.csv").iloc[:, 0]
    y_test = pd.read_csv(INPUT_PATH / "y_test.csv").iloc[:, 0]
    return df, X_train, X_test, y_train, y_test

# ============================================================================
# PART 7: MAIN EXECUTION
# ============================================================================

def main():
    start_time = time.time()
    
    print_header("🛡️  UEBA PROTOTYPE - ML-BASED INSIDER THREAT DETECTION")
    print_info(f"Dataset folder: {BASE_PATH}")
    
    # Check if data exists
    if data_exists():
        print_section("Loading Existing Data (Skipping Generation)")
        df, X_train, X_test, y_train, y_test = load_existing_data()
        print_success(f"Loaded raw data: {len(df):,} records")
        print_success(f"Training set: {len(X_train):,} | Testing set: {len(X_test):,}")
    else:
        # Step 1: Generate Mock Data
        print_section("Step 1: Generating Mock Data")
        generator = MockDataGenerator(n_users=500, n_days=90, n_records_per_day=50)
        df = generator.generate_baseline_behavior()
        df = generator.inject_anomalies(df, anomaly_percentage=2)
        generator.save_raw_data(df)
        print_data("Total records", f"{len(df):,}")
        print_data("Anomalies", f"{df['anomaly_label'].sum():,} ({df['anomaly_label'].mean()*100:.2f}%)")
        
        # Step 2: Feature Engineering
        print_section("Step 2: Engineering Features")
        df = FeatureEngineer.engineer_features(df)
        feature_cols = [col for col in df.columns if col not in 
                       ['timestamp', 'user_id', 'anomaly_label']]
        print_success(f"Created {len(feature_cols)} features")
        
        # Step 3: Prepare Data
        print_section("Step 3: Preparing Training/Testing Data")
        X = df[feature_cols].fillna(0)
        y = df['anomaly_label']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print_success(f"Training set: {len(X_train):,} records")
        print_success(f"Testing set: {len(X_test):,} records")
        
        X_train.to_csv(INPUT_PATH / "X_train.csv", index=False)
        X_test.to_csv(INPUT_PATH / "X_test.csv", index=False)
        y_train.to_csv(INPUT_PATH / "y_train.csv", index=False)
        y_test.to_csv(INPUT_PATH / "y_test.csv", index=False)
        print_success("Saved engineered features to input folder")
    
    feature_cols = list(X_train.columns)
    
    # Step 4: Train Model
    print_section("Step 4: Training Model")
    model = UEBAModel()
    model.train(X_train, y_train)
    print_success("Isolation Forest ✓")
    print_success("Random Forest Classifier ✓")
    
    # Step 5: Make Predictions
    print_section("Step 5: Generating Predictions")
    results = model.predict(X_test)
    print_success("Predictions generated")
    
    # Step 6: Model Evaluation
    print_section("Step 6: Model Evaluation")
    
    print_info("ISOLATION FOREST PERFORMANCE:")
    if_pred = (results['if_prediction'] == -1).astype(int)
    if_report = classification_report(y_test, if_pred, output_dict=True)
    print(classification_report(y_test, if_pred, 
                              target_names=['Normal', 'Anomaly'],
                              digits=4))
    
    print_info("ENSEMBLE MODEL PERFORMANCE:")
    ensemble_pred = results['is_anomaly']
    ensemble_report = classification_report(y_test, ensemble_pred, output_dict=True)
    print(classification_report(y_test, ensemble_pred,
                              target_names=['Normal', 'Anomaly'],
                              digits=4))
    
    roc_score = roc_auc_score(y_test, results['ensemble_score'])
    print_success(f"ROC-AUC Score: {roc_score:.4f}")
    
    # Step 7: Create Visualizations
    print_section("Step 7: Creating Visualizations")
    create_visualizations(results, y_test, feature_cols, X_test, model)
    
    # Step 8: FAST Explainability (Using Feature Importance)
    print_section("Step 8: Fast Explanations (Feature Importance Based)")
    print_warning("Using FAST explanation method (100x faster than SHAP)")
    print_info("Computing feature importance explanations...")
    
    explainer = FastExplainer(model, feature_cols)
        # Create a simple list of user_ids aligned with X_test rows.
    # If X_test index contains original df indices this maps them; if not, you should instead
    # keep the original index in a column before train_test_split (see note below).
    user_ids = X_test.index.map(lambda i: df.iloc[i]['user_id']).to_list()
    explanations_df = explainer.create_explanation_df(
        results['ensemble_score'],
        user_ids,
        X_test
    )

    print_success(f"Generated explanations for top {len(explanations_df)} anomalies")
    
    # Step 9: Generate Alerts
    print_section("Step 9: Generating Alerts")
    
    user_ids = X_test.index.map(lambda i: df.iloc[i]['user_id'])
    
    alerts = pd.DataFrame({
        'user_id': user_ids,
        'anomaly_score': results['ensemble_score'],
        'is_anomaly': results['is_anomaly'],
        'risk_level': pd.cut(results['ensemble_score'], 
                            bins=[0, 0.33, 0.66, 1.0],
                            labels=['🟢 Low', '🟡 Medium', '🔴 High'])
    })
    
    top_alerts = alerts.nlargest(10, 'anomaly_score')
    
    print_info("\n🚨 TOP 10 SUSPICIOUS USERS:\n")
    for idx, row in top_alerts.iterrows():
        print(f"  {Colors.BOLD}{Colors.RED}#{row['user_id']:3d}{Colors.ENDC} | Score: {row['anomaly_score']:.4f} | {row['risk_level']}")
    
    # Step 10: Save Outputs
    print_section("Step 10: Saving All Outputs")
    
    alerts.to_csv(OUTPUT_PATH / "alerts.csv", index=False)
    print_success("Saved: alerts.csv")
    
    top_alerts.to_csv(OUTPUT_PATH / "top_alerts.csv", index=False)
    print_success("Saved: top_alerts.csv")
    
    predictions_df = pd.DataFrame({
        'user_id': user_ids,
        'isolation_forest_score': results['if_score'],
        'ensemble_score': results['ensemble_score'],
        'is_anomaly': results['is_anomaly'],
        'actual_label': y_test.values,
        'risk_level': pd.cut(results['ensemble_score'],
                            bins=[0, 0.33, 0.66, 1.0],
                            labels=['Low', 'Medium', 'High'])
    })
    predictions_df.to_csv(OUTPUT_PATH / "predictions.csv", index=False)
    print_success("Saved: predictions.csv")
    
    # Save Fast Explanations
    explanations_df.to_csv(OUTPUT_PATH / "fast_explanations.csv", index=False)
    print_success("Saved: fast_explanations.csv (⚡ FAST method - Feature Importance Based)")
    
    # Display explanations
    print_section("Top Anomaly Explanations (FAST Method)")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)
    
    for idx, row in explanations_df.head(10).iterrows():
        print(f"\n{Colors.BOLD}{Colors.RED}ANOMALY #{idx+1}{Colors.ENDC}")
        print(f"  User ID: {Colors.BOLD}{row['user_id']}{Colors.ENDC}")
        print(f"  Score: {Colors.BOLD}{row['anomaly_score']}{Colors.ENDC}")
        print(f"  Threat Level: {Colors.BOLD}{row['threat_level']}{Colors.ENDC}")
        print(f"  Top Threats: {Colors.BOLD}{row['threat_description']}{Colors.ENDC}")
        print(f"  Feature 1: {row['top_feature_1']:25s} (Importance: {row['feature_1_importance']}, Value: {row['feature_1_value']})")
        print(f"  Feature 2: {row['top_feature_2']:25s} (Importance: {row['feature_2_importance']}, Value: {row['feature_2_value']})")
        print(f"  Feature 3: {row['top_feature_3']:25s} (Importance: {row['feature_3_importance']}, Value: {row['feature_3_value']})")
        print(f"  Action: {Colors.BOLD}{Colors.YELLOW}{row['action_required']}{Colors.ENDC}")
    
    # Save metrics
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'total_records': len(df),
        'training_records': len(X_train),
        'testing_records': len(X_test),
        'anomalies_in_dataset': int(df['anomaly_label'].sum()),
        'anomaly_percentage': float(df['anomaly_label'].mean() * 100),
        'roc_auc_score': float(roc_score),
        'isolation_forest_metrics': if_report,
        'ensemble_metrics': ensemble_report,
        'execution_time_seconds': round(time.time() - start_time, 2)
    }
    
    with open(OUTPUT_PATH / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    print_success("Saved: metrics.json")
    
    # Step 11: Feature Importance Ranking
    print_section("Step 11: Feature Importance Ranking")
    importances = model.rf_classifier.feature_importances_
    top_features_idx = np.argsort(importances)[-20:][::-1]
    
    for rank, idx in enumerate(top_features_idx, 1):
        bar_length = int(importances[idx] * 100)
        bar = "█" * bar_length + "░" * (100 - bar_length)
        print_info(f"{rank:2d}. {feature_cols[idx]:30s} {Colors.BOLD}[{bar}]{Colors.ENDC} {importances[idx]:.4f}")
    
    print_header("✅ PROTOTYPE EXECUTION COMPLETE")
    print_info(f"Total execution time: {Colors.BOLD}{time.time() - start_time:.2f} seconds{Colors.ENDC}")
    print_info(f"Output files: {Colors.BOLD}{OUTPUT_PATH}{Colors.ENDC}")
    print_info(f"Visualizations: {Colors.BOLD}{VIZ_PATH}{Colors.ENDC}")
    print_success("All outputs saved successfully!")
    
    return model, alerts, df, explanations_df

if __name__ == "__main__":
    model, alerts, df, explanations = main()