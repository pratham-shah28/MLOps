# Import necessary libraries and modules
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from src.lab import (
    load_training_data,
    preprocess_training_data,
    train_and_save_classifier,
    predict_with_saved_model,
)


default_args = {
    'owner': 'your_name',
    'start_date': datetime(2025, 1, 15),
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'Airflow_Lab1',
    default_args=default_args,
    description='Classification pipeline: preprocess -> train RF -> predict',
    schedule_interval=None,
    catchup=False,
)

# Locating training CSV
load_training_data_task = PythonOperator(
    task_id='load_training_data',
    python_callable=load_training_data,
    dag=dag,
)

# Fitting imputer+scaler, saving artifacts, persisting matrices to disk
preprocess_training_data_task = PythonOperator(
    task_id='preprocess_training_data',
    python_callable=preprocess_training_data,
    op_args=[load_training_data_task.output],
    dag=dag,
)

# Training RandomForest and save model/model.sav
train_and_save_classifier_task = PythonOperator(
    task_id='train_and_save_classifier',
    python_callable=train_and_save_classifier,
    op_args=[preprocess_training_data_task.output, "model.sav"],
    dag=dag,
)

# Predicting on data/test.csv with saved artifacts
predict_with_saved_model_task = PythonOperator(
    task_id='predict_with_saved_model',
    python_callable=predict_with_saved_model,
    op_args=[train_and_save_classifier_task.output, preprocess_training_data_task.output],
    dag=dag,
)

# Order
load_training_data_task >> preprocess_training_data_task >> train_and_save_classifier_task >> predict_with_saved_model_task

if __name__ == "__main__":
    dag.cli()
