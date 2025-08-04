import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import pickle

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import (
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
    ColumnDriftMetric,
    ClassificationQualityMetric,
    ClassificationClassBalance,
    ClassificationConfusionMatrix
)
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestNumberOfColumnsWithMissingValues,
    TestNumberOfRowsWithMissingValues,
    TestNumberOfConstantColumns,
    TestNumberOfDuplicatedColumns,
    TestColumnsType,
    TestNumberOfDriftedColumns
)

import psycopg2
from psycopg2.extras import RealDictCursor
import sqlite3

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelMonitor:
    """Model monitoring class using Evidently AI"""
    
    def __init__(self, reference_data_path: str, db_config: Optional[Dict] = None):
        """Initialize the monitor with reference data"""
        self.reference_data = pd.read_csv(reference_data_path)
        self.db_config = db_config
        self.column_mapping = self._create_column_mapping()
        
        # Create monitoring directory
        self.monitoring_dir = Path("monitoring")
        self.monitoring_dir.mkdir(exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _create_column_mapping(self) -> ColumnMapping:
        """Create column mapping for Evidently"""
        numerical_features = ['age', 'bmi', 'sleep_hours']
        categorical_features = [
            'family_history', 'diet', 'exercise_frequency', 
            'smoking_status', 'alcohol_consumption'
        ]
        
        return ColumnMapping(
            target='risk_level',
            prediction='prediction',
            numerical_features=numerical_features,
            categorical_features=categorical_features
        )
    
    def _init_database(self):
        """Initialize SQLite database for storing monitoring results"""
        db_path = self.monitoring_dir / "monitoring.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create table for drift metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS drift_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                dataset_drift REAL,
                n_drifted_columns INTEGER,
                share_drifted_columns REAL,
                drift_details TEXT
            )
        """)
        
        # Create table for model performance
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                accuracy REAL,
                precision_macro REAL,
                recall_macro REAL,
                f1_macro REAL,
                n_predictions INTEGER
            )
        """)
        
        # Create table for prediction logs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                input_data TEXT,
                prediction INTEGER,
                probabilities TEXT,
                model_run_id TEXT
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    def load_prediction_logs(self, start_date: Optional[datetime] = None, 
                           end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Load prediction logs from JSON files"""
        logs_dir = Path("logs")
        predictions_file = logs_dir / "predictions.jsonl"
        
        if not predictions_file.exists():
            logger.warning("No prediction logs found")
            return pd.DataFrame()
        
        # Read JSONL file
        logs = []
        with open(predictions_file, 'r') as f:
            for line in f:
                try:
                    log_entry = json.loads(line.strip())
                    logs.append(log_entry)
                except json.JSONDecodeError:
                    continue
        
        if not logs:
            logger.warning("No valid prediction logs found")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df_logs = pd.DataFrame(logs)
        df_logs['timestamp'] = pd.to_datetime(df_logs['timestamp'])
        
        # Filter by date range if provided
        if start_date:
            df_logs = df_logs[df_logs['timestamp'] >= start_date]
        if end_date:
            df_logs = df_logs[df_logs['timestamp'] <= end_date]
        
        return df_logs
    
    def prepare_current_data(self, prediction_logs: pd.DataFrame) -> pd.DataFrame:
        """Prepare current data from prediction logs"""
        if prediction_logs.empty:
            return pd.DataFrame()
        
        # Extract input features
        input_features = []
        for _, row in prediction_logs.iterrows():
            features = row['input'].copy()
            features['prediction'] = row['prediction']
            
            # Add probabilities
            probs = row['probabilities']
            features['prob_low'] = probs[0]
            features['prob_medium'] = probs[1]
            features['prob_high'] = probs[2]
            
            input_features.append(features)
        
        current_data = pd.DataFrame(input_features)
        
        # Apply same preprocessing as reference data
        categorical_mappings = {
            'family_history': {'No': 0, 'Yes': 1},
            'diet': {'Healthy': 0, 'Moderate': 1, 'Unhealthy': 2},
            'exercise_frequency': {'Daily': 0, 'Monthly': 1, 'Rarely': 2, 'Weekly': 3},
            'smoking_status': {'Current': 0, 'Former': 1, 'Never': 2},
            'alcohol_consumption': {'Heavy': 0, 'Light': 1, 'Moderate': 2, 'None': 3}
        }
        
        for col, mapping in categorical_mappings.items():
            if col in current_data.columns:
                current_data[col] = current_data[col].map(mapping)
        
        return current_data
    
    def calculate_drift_metrics(self, current_data: pd.DataFrame) -> Dict:
        """Calculate data drift metrics"""
        if current_data.empty:
            logger.warning("No current data available for drift calculation")
            return {}
        
        # Create drift report
        drift_report = Report(metrics=[
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
        ])
        
        # Add individual column drift metrics
        for col in self.reference_data.columns:
            if col in current_data.columns and col != 'risk_level':
                drift_report.metrics.append(ColumnDriftMetric(column_name=col))
        
        # Run the report
        drift_report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        # Extract metrics
        drift_results = drift_report.as_dict()
        
        # Parse key metrics
        dataset_drift = drift_results['metrics'][0]['result']['dataset_drift']
        n_drifted_columns = drift_results['metrics'][0]['result']['number_of_drifted_columns']
        share_drifted_columns = drift_results['metrics'][0]['result']['share_of_drifted_columns']
        
        metrics = {
            'dataset_drift': dataset_drift,
            'n_drifted_columns': n_drifted_columns,
            'share_drifted_columns': share_drifted_columns,
            'drift_details': drift_results
        }
        
        # Save report
        report_path = self.monitoring_dir / f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        drift_report.save_html(str(report_path))
        logger.info(f"Drift report saved to {report_path}")
        
        return metrics
    
    def calculate_model_performance(self, current_data: pd.DataFrame) -> Dict:
        """Calculate model performance metrics (when ground truth is available)"""
        # For now, we'll simulate this since we don't have ground truth in production
        # In a real scenario, you would collect ground truth labels periodically
        
        if current_data.empty or 'risk_level' not in current_data.columns:
            logger.warning("No ground truth available for performance calculation")
            return {}
        
        performance_report = Report(metrics=[
            ClassificationQualityMetric(),
            ClassificationClassBalance(),
            ClassificationConfusionMatrix()
        ])
        
        performance_report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        # Extract metrics
        perf_results = performance_report.as_dict()
        
        # Parse performance metrics
        quality_metrics = perf_results['metrics'][0]['result']['current']
        
        metrics = {
            'accuracy': quality_metrics.get('accuracy', 0),
            'precision_macro': quality_metrics.get('precision', {}).get('macro avg', 0),
            'recall_macro': quality_metrics.get('recall', {}).get('macro avg', 0),
            'f1_macro': quality_metrics.get('f1-score', {}).get('macro avg', 0),
            'n_predictions': len(current_data)
        }
        
        # Save report
        report_path = self.monitoring_dir / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        performance_report.save_html(str(report_path))
        logger.info(f"Performance report saved to {report_path}")
        
        return metrics
    
    def run_data_quality_tests(self, current_data: pd.DataFrame) -> Dict:
        """Run data quality tests"""
        if current_data.empty:
            return {}
        
        # Create test suite
        data_quality_suite = TestSuite(tests=[
            TestNumberOfColumnsWithMissingValues(),
            TestNumberOfRowsWithMissingValues(),
            TestNumberOfConstantColumns(),
            TestNumberOfDuplicatedColumns(),
            TestColumnsType(),
            TestNumberOfDriftedColumns()
        ])
        
        # Run tests
        data_quality_suite.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        # Save test results
        test_path = self.monitoring_dir / f"data_quality_tests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        data_quality_suite.save_html(str(test_path))
        logger.info(f"Data quality tests saved to {test_path}")
        
        # Extract test results
        test_results = data_quality_suite.as_dict()
        
        return test_results
    
    def save_metrics_to_db(self, drift_metrics: Dict, performance_metrics: Dict):
        """Save metrics to database"""
        db_path = self.monitoring_dir / "monitoring.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        # Save drift metrics
        if drift_metrics:
            cursor.execute("""
                INSERT INTO drift_metrics 
                (timestamp, dataset_drift, n_drifted_columns, share_drifted_columns, drift_details)
                VALUES (?, ?, ?, ?, ?)
            """, (
                timestamp,
                float(drift_metrics.get('dataset_drift', 0)),
                int(drift_metrics.get('n_drifted_columns', 0)),
                float(drift_metrics.get('share_drifted_columns', 0)),
                json.dumps(drift_metrics.get('drift_details', {}))
            ))
        
        # Save performance metrics
        if performance_metrics:
            cursor.execute("""
                INSERT INTO model_performance 
                (timestamp, accuracy, precision_macro, recall_macro, f1_macro, n_predictions)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                timestamp,
                float(performance_metrics.get('accuracy', 0)),
                float(performance_metrics.get('precision_macro', 0)),
                float(performance_metrics.get('recall_macro', 0)),
                float(performance_metrics.get('f1_macro', 0)),
                int(performance_metrics.get('n_predictions', 0))
            ))
        
        conn.commit()
        conn.close()
        logger.info("Metrics saved to database")
    
    def generate_monitoring_dashboard(self) -> str:
        """Generate a simple HTML monitoring dashboard"""
        db_path = self.monitoring_dir / "monitoring.db"
        conn = sqlite3.connect(db_path)
        
        # Get recent drift metrics
        drift_df = pd.read_sql_query("""
            SELECT * FROM drift_metrics 
            ORDER BY timestamp DESC 
            LIMIT 10
        """, conn)
        
        # Get recent performance metrics
        perf_df = pd.read_sql_query("""
            SELECT * FROM model_performance 
            ORDER BY timestamp DESC 
            LIMIT 10
        """, conn)
        
        conn.close()
        
        # Create simple HTML dashboard
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Monitoring Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .metric-card {{ 
                    background: white; padding: 20px; margin: 20px 0; 
                    border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
                }}
                .metric-value {{ font-size: 2em; font-weight: bold; color: #007bff; }}
                .metric-label {{ color: #666; margin-top: 5px; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f8f9fa; }}
                .alert {{ padding: 15px; margin: 20px 0; border-radius: 4px; }}
                .alert-warning {{ background-color: #fff3cd; border: 1px solid #ffeaa7; }}
                .alert-danger {{ background-color: #f8d7da; border: 1px solid #f5c6cb; }}
                .alert-success {{ background-color: #d4edda; border: 1px solid #c3e6cb; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Model Monitoring Dashboard</h1>
                <p>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <div class="metric-card">
                    <h2>Data Drift Status</h2>
                    {"<div class='alert alert-warning'>Warning: Data drift detected!</div>" if not drift_df.empty and drift_df.iloc[0]['dataset_drift'] else "<div class='alert alert-success'>No data drift detected</div>"}
                    
                    {f"<div class='metric-value'>{drift_df.iloc[0]['share_drifted_columns']:.2%}</div>" if not drift_df.empty else "<div class='metric-value'>N/A</div>"}
                    <div class="metric-label">Share of Drifted Columns</div>
                </div>
                
                <div class="metric-card">
                    <h2>Recent Drift Metrics</h2>
                    {drift_df.to_html(classes='table', table_id='drift-table') if not drift_df.empty else "<p>No drift data available</p>"}
                </div>
                
                <div class="metric-card">
                    <h2>Model Performance</h2>
                    {f"<div class='metric-value'>{perf_df.iloc[0]['accuracy']:.3f}</div>" if not perf_df.empty else "<div class='metric-value'>N/A</div>"}
                    <div class="metric-label">Latest Accuracy</div>
                </div>
                
                <div class="metric-card">
                    <h2>Recent Performance Metrics</h2>
                    {perf_df.to_html(classes='table', table_id='perf-table') if not perf_df.empty else "<p>No performance data available</p>"}
                </div>
            </div>
        </body>
        </html>
        """
        
        dashboard_path = self.monitoring_dir / "dashboard.html"
        with open(dashboard_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Monitoring dashboard saved to {dashboard_path}")
        return str(dashboard_path)
    
    def run_monitoring_cycle(self, days_back: int = 7):
        """Run a complete monitoring cycle"""
        logger.info("Starting monitoring cycle")
        
        # Load recent prediction logs
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        prediction_logs = self.load_prediction_logs(start_date, end_date)
        
        if prediction_logs.empty:
            logger.warning("No recent prediction logs found")
            return
        
        logger.info(f"Found {len(prediction_logs)} predictions in the last {days_back} days")
        
        # Prepare current data
        current_data = self.prepare_current_data(prediction_logs)
        
        # Calculate drift metrics
        drift_metrics = self.calculate_drift_metrics(current_data)
        
        # Calculate performance metrics (if ground truth available)
        performance_metrics = self.calculate_model_performance(current_data)
        
        # Run data quality tests
        quality_results = self.run_data_quality_tests(current_data)
        
        # Save metrics to database
        self.save_metrics_to_db(drift_metrics, performance_metrics)
        
        # Generate dashboard
        dashboard_path = self.generate_monitoring_dashboard()
        
        logger.info(f"Monitoring cycle completed. Dashboard available at: {dashboard_path}")
        
        return {
            'drift_metrics': drift_metrics,
            'performance_metrics': performance_metrics,
            'quality_results': quality_results,
            'dashboard_path': dashboard_path
        }

def main():
    """Main monitoring function"""
    # Initialize monitor with reference data
    reference_data_path = "/workspace/Data/raw/synthetic_prostate_cancer_risk.csv"
    
    if not Path(reference_data_path).exists():
        logger.error(f"Reference data not found at {reference_data_path}")
        return
    
    monitor = ModelMonitor(reference_data_path)
    
    # Run monitoring cycle
    results = monitor.run_monitoring_cycle(days_back=7)
    
    if results:
        logger.info("Monitoring completed successfully")
        if results['drift_metrics'].get('dataset_drift'):
            logger.warning("Data drift detected! Check the monitoring dashboard for details.")
    else:
        logger.info("No monitoring data available")

if __name__ == "__main__":
    main()