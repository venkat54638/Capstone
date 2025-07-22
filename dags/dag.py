from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.dates import days_ago
import subprocess
import os

# Define the path to your scripts directory within the Airflow container
AIRFLOW_HOME = os.getenv('AIRFLOW_HOME', '/opt/airflow')
SCRIPTS_DIR = os.path.join(AIRFLOW_HOME, 'scripts')

def execute_main_script():
    """
    This function navigates to the scripts directory and executes the main.py script.
    """
    script_path = os.path.join(SCRIPTS_DIR, 'main.py')
    print(f"Executing script: {script_path}")
    
    # Using subprocess.run to execute the script.
    # check=True will raise an exception if the script returns a non-zero exit code,
    # which will cause the Airflow task to fail as expected.
    result = subprocess.run(
        ['python', script_path],
        capture_output=True,
        text=True,
        check=True
    )
    print("Script STDOUT:")
    print(result.stdout)

with DAG(
    dag_id='execute_single_main_script_pipeline',
    default_args={'owner': 'airflow'},
    start_date=days_ago(1),
    schedule_interval=None,
    description='A simple DAG to run a single main.py ML script',
    catchup=False,
    tags=['simple-pipeline'],
) as dag:

    start = DummyOperator(task_id='start')

    run_end_to_end_pipeline = PythonOperator(
        task_id='run_end_to_end_pipeline',
        python_callable=execute_main_script
    )

    end = DummyOperator(task_id='end')

    start >> run_end_to_end_pipeline >> end