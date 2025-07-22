from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import timedelta

import boto3
import numpy as np
import pandas as pd
from airflow.datasets import Dataset
from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.utils.dates import days_ago

# Imports for Data Drift Check
from evidently.metric_preset import DataDriftPreset
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report

# Define a Dataset representing the source Redshift table.
redshift_table_dataset = Dataset("redshift://lead/dev/sale")

# --- Configuration Variables ---
S3_BUCKET = 'dag543'
S3_REFERENCE_DATA_KEY = 'reference/training_data.csv'
S3_MAIN_SCRIPT_KEY = 'scripts/full_pipeline.py'
S3_ARTIFACT_PREFIX = 'artifacts'
MLFLOW_TRACKING_URI = 'arn:aws:sagemaker:ap-south-1:152320433616:mlflow-tracking-server/mlflow'
MODEL_NAME = 'LeadConversionModel'

# --- Default Arguments for DAG ---
default_args = {
    'owner': 'Anil Kumar',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# --- DAG Definition ---
with DAG(
    'lead_conversion_master_pipeline',
    default_args=default_args,
    start_date=days_ago(1),
    schedule=[redshift_table_dataset],  # Event-driven schedule
    catchup=False,
    tags=['mlops', 'aws', 'evidently', 'drift_detection'],
) as dag:

    install_deps = BashOperator(
        task_id='install_dependencies',
        bash_command='pip install pandas boto3 "evidently>=0.4.0" scikit-learn numpy xgboost',
    )

    def fetch_data_from_redshift(**kwargs):
        """Fetches data from Redshift and pushes the S3 path to XComs."""
        print("--- Fetching data from Redshift ---")
        region = 'ap-south-1'
        workgroup_name = 'lead'
        database_name = 'dev'
        secret_arn = 'arn:aws:secretsmanager:ap-south-1:152320433616:secret:lead-yvU3pC'
        sql = 'SELECT * FROM sale'

        client = boto3.client('redshift-data', region_name=region)
        response = client.execute_statement(
            WorkgroupName=workgroup_name, Database=database_name, SecretArn=secret_arn, Sql=sql
        )
        statement_id = response['Id']

        while True:
            desc = client.describe_statement(Id=statement_id)
            if desc['Status'] in ['FINISHED', 'FAILED', 'ABORTED']:
                break
            time.sleep(2)

        if desc['Status'] != 'FINISHED':
            raise Exception(f"Redshift query failed: {desc.get('Error')}")

        result = client.get_statement_result(Id=statement_id)
        columns = [col['name'] for col in result['ColumnMetadata']]
        data = [[list(col.values())[0] for col in row] for row in result['Records']]
        df = pd.DataFrame(data, columns=columns)

        ds = kwargs['ds']
        s3_path = f'raw/lead_scoring_{ds}.csv'
        with tempfile.NamedTemporaryFile(mode='w', suffix=".csv") as tmp:
            df.to_csv(tmp.name, index=False)
            boto3.client('s3').upload_file(tmp.name, S3_BUCKET, s3_path)
        print(f"Data uploaded to s3://{S3_BUCKET}/{s3_path}")
        kwargs['ti'].xcom_push(key='new_data_s3_key', value=s3_path)

    def check_data_drift(**kwargs):
        """Generates drift reports using explicit column mapping and robust type handling."""
        ti = kwargs['ti']
        new_data_s3_key = ti.xcom_pull(key='new_data_s3_key', task_ids='fetch_from_redshift')
        if not new_data_s3_key:
            raise ValueError("Could not find S3 key for new data.")

        s3 = boto3.client('s3')
        with tempfile.TemporaryDirectory() as tmpdir:
            new_data_path = os.path.join(tmpdir, 'new_data.csv')
            ref_data_path = os.path.join(tmpdir, 'ref_data.csv')
            s3.download_file(S3_BUCKET, new_data_s3_key, new_data_path)
            s3.download_file(S3_BUCKET, S3_REFERENCE_DATA_KEY, ref_data_path)

            new_data = pd.read_csv(new_data_path)
            ref_data = pd.read_csv(ref_data_path)

            def standardize_cols(df):
                df.columns = [str(col).strip().lower().replace(' ', '_') for col in df.columns]
                return df
            new_data, ref_data = standardize_cols(new_data), standardize_cols(ref_data)

            if 'converted' in ref_data.columns:
                ref_data = ref_data.drop(columns=['converted'])
            if 'converted' in new_data.columns:
                new_data = new_data.drop(columns=['converted'])

            shared_columns = list(set(ref_data.columns) & set(new_data.columns))
            if not shared_columns:
                raise ValueError("CRITICAL: No common columns found between reference and new data.")

            # --- Robustly define column types to prevent TypeError ---
            numerical_cols = ref_data[shared_columns].select_dtypes(include=np.number).columns.tolist()
            categorical_cols = ref_data[shared_columns].select_dtypes(exclude=np.number).columns.tolist()

            clean_numerical_cols = []
            for col in numerical_cols:
                ref_data[col] = pd.to_numeric(ref_data[col], errors='coerce')
                new_data[col] = pd.to_numeric(new_data[col], errors='coerce')
                if ref_data[col].isnull().any() or new_data[col].isnull().any():
                     categorical_cols.append(col)
                else:
                     clean_numerical_cols.append(col)

            column_mapping = ColumnMapping(
                numerical_features=clean_numerical_cols,
                categorical_features=categorical_cols
            )

            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=ref_data, current_data=new_data, column_mapping=column_mapping)

            ds = kwargs['ds']
            html_report_path = os.path.join(tmpdir, f"drift_report_{ds}.html")
            report.save_html(html_report_path)
            html_s3_key = f'reports/drift_report_{ds}.html'
            s3.upload_file(html_report_path, S3_BUCKET, html_s3_key)
            print(f"HTML drift report saved to: s3://{S3_BUCKET}/{html_s3_key}")

            drift_detected = report.as_dict()['metrics'][0]['result']['dataset_drift']
            print(f"Drift Detected: {drift_detected}")
            ti.xcom_push(key='drift_detected', value=drift_detected)

    def branch_on_drift(**kwargs):
        """Decides which path to take based on the drift result."""
        drift_detected = kwargs['ti'].xcom_pull(key='drift_detected', task_ids='check_drift')
        return 'retrain_model' if drift_detected else 'skip_retraining'

    def run_ml_pipeline_locally(**kwargs):
        """Downloads and executes the ML training script."""
        ti = kwargs['ti']
        ds = kwargs['ds']
        s3_input_key = ti.xcom_pull(key='new_data_s3_key', task_ids='fetch_from_redshift')

        with tempfile.TemporaryDirectory() as tmpdir:
            local_script_path = os.path.join(tmpdir, "pipeline.py")
            local_data_path = os.path.join(tmpdir, 'data.csv')

            s3 = boto3.client('s3')
            s3.download_file(S3_BUCKET, S3_MAIN_SCRIPT_KEY, local_script_path)
            s3.download_file(S3_BUCKET, s3_input_key, local_data_path)

            script_args = [
                sys.executable, local_script_path,
                "--data-path", local_data_path,
                "--s3-bucket", S3_BUCKET,
                "--s3-artifact-prefix", S3_ARTIFACT_PREFIX,
                "--mlflow-uri", MLFLOW_TRACKING_URI,
                "--model-name", MODEL_NAME,
                "--date", ds,
            ]

            process = subprocess.run(script_args, capture_output=True, text=True, check=False)
            print("--- Script STDOUT ---\n", process.stdout)
            print("--- Script STDERR ---\n", process.stderr)
            if process.returncode != 0:
                raise Exception(f"ML script failed with return code {process.returncode}")

    # --- Task Definitions ---
    fetch_data_task = PythonOperator(
        task_id='fetch_from_redshift',
        python_callable=fetch_data_from_redshift,
    )

    check_drift_task = PythonOperator(
        task_id='check_drift',
        python_callable=check_data_drift,
    )

    branching_task = BranchPythonOperator(
        task_id='branch_on_drift',
        python_callable=branch_on_drift,
    )

    retrain_model_task = PythonOperator(
        task_id='retrain_model',
        python_callable=run_ml_pipeline_locally,
    )

    skip_retraining_task = DummyOperator(task_id='skip_retraining')

    end_pipeline_task = DummyOperator(
        task_id='end_pipeline',
        trigger_rule='none_failed_min_one_success',
    )

    # --- DAG Flow Definition ---
    install_deps >> fetch_data_task >> check_drift_task >> branching_task
    branching_task >> [retrain_model_task, skip_retraining_task]
    retrain_model_task >> end_pipeline_task
    skip_retraining_task >> end_pipeline_task 