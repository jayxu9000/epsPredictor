from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.ssh.operators.ssh import SSHOperator

# Default arguments for the DAG
default_args = {
    'owner': 'jay',
    'depends_on_past': False,
    'start_date': datetime(2024, 3, 28),
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'eps_etl_and_visuals',
    default_args=default_args,
    description='ETL pipeline + visuals for EPS data using Spark and Airflow',
    schedule_interval='@daily',
    catchup=False,
    max_active_runs=1,
) as dag:

    # 1) Run ETL
    etl_command = """
    export JAVA_HOME=/opt/java/openjdk;
    /opt/spark/bin/spark-submit \
      --master spark://spark-leader:7077 \
      /opt/spark/work-dir/etl.py
    """
    run_spark_etl = SSHOperator(
        task_id='run_spark_etl',
        ssh_conn_id='spark_ssh',
        command=etl_command,
        get_pty=True,
        cmd_timeout=None,
    )

    # 2) Run visuals.py after ETL completes
    visuals_command = """
    export JAVA_HOME=/opt/java/openjdk;
    /opt/spark/bin/spark-submit \
      --master spark://spark-leader:7077 \
      /opt/spark/work-dir/visuals.py
    """
    run_spark_visuals = SSHOperator(
        task_id='run_spark_visuals',
        ssh_conn_id='spark_ssh',
        command=visuals_command,
        get_pty=True,
        cmd_timeout=None,
    )

    # set the dependency: etl → visuals
    run_spark_etl >> run_spark_visuals
