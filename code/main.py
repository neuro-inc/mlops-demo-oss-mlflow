import platform
import mlflow
from argparse import ArgumentParser

client = mlflow.tracking.MlflowClient()


def run():
    mlflow.start_run()
    mlflow.log_param('alpha', 0.5)
    mlflow.log_metric('rmse', 0.789)
    mlflow.set_tag('mlflow.version', mlflow.__version__)
    mlflow.set_tag('python.version', platform.python_version())
    mlflow.set_tag('platform.version', platform.system())
    mlflow.set_tag('version.', platform.machine())
    mlflow.end_run()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--experiment_name', dest='experiment_name', help='Experiment name', default='Default', type=str)
    args = parser.parse_args()

    mlflow.set_experiment(args.experiment_name)
    run()