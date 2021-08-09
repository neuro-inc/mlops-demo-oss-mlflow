# MLFlow Tutorial in Neuro MLOps Platform 

This is an adaption of the [NLP FROM SCRATCH: CLASSIFYING NAMES WITH A CHARACTER-LEVEL RNN](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html) tutorial to the [Neu.ro platform](https://neu.ro).

## Quick Start

Sign up at [app.neu.ro](https://app.neu.ro) and setup your local machine according to [instructions](https://docs.neu.ro/).

Good examples could be found in the [repository](https://github.com/amesar/mlflow-examples)
 
Then run:

```shell
pip install -U neuro-cli neuro-flow
neuro login
# build docker image remotely on the platform
neuro-flow build myimage
# prepare storage folders
neuro-flow mkvolumes
# and upload data
neuro-flow upload ALL
# to work with an interactive notebook, hit
neuro-flow run tensorboard
neuro-flow run train
```

See [Help.md](HELP.md) for the detailed Neuro Project Template Reference.


pip install -U neuro-cli neuro-flow
neuro login
neuro-flow build myimage
neuro-flow mkvolumes
neuro-flow upload ALL
neuro-flow run train
neuro-flow run server
# neuro-flow run jupyter
# neuro-flow run tensorboard
# http get localhost:8080 line='Smith'
# python3 rnn/predict Smith
# http get https://job-c623cb5f-36cb-493a-bbc9-1d38e86ce971.jobs.neuro-compute.org.neu.ro/ line='Smith'
# locust -f rnn/test.py --headless -u 10 -r 100 --host https://job-559448e2-2578-458b-83d1-95ba3285c000.jobs.neuro-compute.org.neu.ro


Create a persistent disk for Postgresql, export it as an env variable
Run a Postgresql server:
```shell
neuro disk create 1G --timeout-unused 30d --name mlops-demo-oss-dogs-postgres
export MLFLOW_STORAGE=mlops-demo-oss-dogs-postgres
neuro-flow run postgres
```

Run an MLFlow server for experiment and model tracking. In this setup, we imply personal use of MLFlow server (each user will connect to their own server).

```shell
neuro-flow run mlflow
```

Set up the variables needed to run the loads in your cluster that were provided by the Neu.ro team:

```shell
export MLFLOW_URI=https://demo-oss-names-mlflow-server.jobs.default.org.neu.ro
```

neuro-flow run create_pipeline --param mlflow_storage $MLFLOW_STORAGE --param mlflow_uri $MLFLOW_URI
