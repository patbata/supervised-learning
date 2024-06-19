# import parsing and os packages
import logging
import os

# import data packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import self helper functions
import src.model.modeling as modeling
from src.utils.parsing_utils import (
    config_parser,
    yaml_config_reader,
    set_dataset,
)

# set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)


# run initial tuning
def run_tuning(
        model_name,
        config,
        dataset_object):
    """
    Runs tuning for models

    :param model_name: name of model to run
    :param config: model config file
    :param dataset_object: dataset object
    """
    # store model params
    model_config = config[model_name]

    # check on scaler
    if 'scaler' in model_config:
        scaler_needed = model_config['scaler']
    else:
        scaler_needed = None

    # change directory for saving plots
    original_path = os.getcwd()
    result_path = f"results/{config['data']}/tuning/{model_name}/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    os.chdir(result_path)

    # set up model
    model_to_test = {'dt': 'DecisionTree',
                     'nn': 'NeuralNetwork',
                     'boost': 'BoostedDT',
                     'svm': 'SVM',
                     'knn': 'KNN'}[model_name]
    logging.info(f"Model: {model_to_test}")

    # # run validation for model
    for param, values in model_config["hyperparams"].items():# fix values
        try:
            values = eval(values)
        except:
            values = values

        # plot validation curve
        modeling.plot_validation_curve(model_name=model_to_test,
                                       dataset_object=dataset_object,
                                       param_name=param,
                                       param_range=values,
                                       k_folds=config['k_folds'],
                                       scoring=config['scoring'],
                                       stratified=config['stratified'],
                                       scaler=scaler_needed,)

    # plot learning curve
    modeling.plot_learning_curve(model_name=model_to_test,
                                 dataset_object=dataset_object,
                                 k_folds=config['k_folds'],
                                 scoring=config['scoring'],
                                 model_params=model_config['params'],
                                 stratified=config['stratified'],
                                 scaler=scaler_needed,)
    # log hyperparams
    logging.info(f"Validation curve on: {list(model_config['hyperparams'].keys())}")
    logging.info(f"Results saved to {result_path}")

    # move back into original path
    os.chdir(original_path)


# run gridsearch
def run_gridsearch(
        model_name,
        config,
        dataset_object
):
    """
    Runs gridsearch for models.

    :param model_name: name of model to run
    :param config: model config file
    :param dataset_object: dataset object
    """
    # store model params
    model_config = config[model_name]

    # check on scaler
    if 'scaler' in model_config:
        scaler_needed = model_config['scaler']
    else:
        scaler_needed = None

    # store model params
    original_path = os.getcwd()

    # change directory for saving plots
    result_path = f"results/{config['data']}/gridsearch/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    os.chdir(result_path)

    # set up model
    model_to_test = {'dt': 'DecisionTree',
                     'nn': 'NeuralNetwork',
                     'boost': 'BoostedDT',
                     'svm': 'SVM',
                     'knn': 'KNN'}[model_name]
    logging.info(f"Model: {model_to_test}")

    # fix dictionary object
    try:
        gridsearch_params = {k: eval(v) for k, v in
                             model_config['hyperparams'].items()}
    except:
        gridsearch_params = model_config['hyperparams']

        # fix dictionary object
        for k, v in gridsearch_params.items():
            if k in ['hidden_layer_sizes', 'n_estimators',
                     'n_neighbors', 'leaf_size', 'p']:
                gridsearch_params[k] = eval(v)

    # run gridsearch for model
    results = modeling.grid_search_results(
                 model_to_test,
                 params=gridsearch_params,
                 dataset_object=dataset_object,
                 k_folds=config['k_folds'],
                 stratified=config['stratified'],
                 scoring=config['scoring'],
                 scaler=scaler_needed,)
    results.to_csv(f"{model_name}_gridsearch_results.csv")

    # log and store results
    logging.info(f"Saved gridsearch results to {result_path}")

    # move back into original path
    os.chdir(original_path)


# run final tuning with results
def run_final_results(
        model_name,
        config,
        dataset_object
):
    """
    Runs final tuning for models.

    :param model_name: name of model to run
    :param config: model config file
    :param dataset_object: dataset object
    """
    # store model params
    model_config = config[model_name]

    # check on scaler
    if 'scaler' in model_config:
        scaler_needed = model_config['scaler']
    else:
        scaler_needed = None

    # store model params
    original_path = os.getcwd()

    # change directory for saving plots
    result_path = f"results/{config['data']}/final/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    os.chdir(result_path)

    # check if summary file exists, if not create it
    final_result_file = f'_final_results_summary_{dataset_object.name}.txt'
    val_method = "StratifiedKFold" if config['stratified'] else "KFold"
    if not os.path.exists(final_result_file):
        with open(final_result_file, "w") as f:
            f.write('-----------------------------------------------------------\n')
            f.write(f"{config['data'].upper()} FINAL RESULTS\n")
            f.write(f"Validation Method: {val_method} ({config['k_folds']} folds)\n")
            f.write(f"Scoring:           {config['scoring']}\n")
            f.write(f"Features:          {', '.join(dataset_object.X_train.columns)}\n")

    # set up model
    model_to_test = {'dt': 'DecisionTree',
                     'nn': 'NeuralNetwork',
                     'boost': 'BoostedDT',
                     'svm': 'SVM',
                     'knn': 'KNN'}[model_name]
    logging.info(f"Model: {model_to_test}")

    # run final tuning for model
    modeling.get_best_model(model_name=model_to_test,
                            model_params=model_config['params'],
                            dataset_object=dataset_object,
                            k_folds=config['k_folds'],
                            stratified=config['stratified'],
                            scoring=config['scoring'],
                            filename=final_result_file,
                            scaler=scaler_needed,
                            )

    # log and store results
    logging.info(f"Saved final results to {result_path}")

    # move back into original path
    os.chdir(original_path)


# run preliminary data analysis
def run_preliminary_analysis(
        dataset_object
):
    """
    Runs preliminary data analysis for dataset.

    :param dataset_object: dataset object
    """
    # change directory for saving plots
    original_path = os.getcwd()
    result_path = f"results/{dataset_object.name}/preliminary/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    os.chdir(result_path)

    # Create distribution plot
    filepath = f'/Users/pbata/Documents/omscs-ml//data/{dataset_object.name}-data/{dataset_object.name}.csv'
    dataset_one = pd.read_csv(filepath, sep=';')
    output = dataset_one[dataset_object.target_variable].value_counts() #for uncleaned
    # output = dataset_object.y.value_counts()  # for cleaned
    output.sort_index().plot(kind='bar')
    plt.title(f"{dataset_object.name.title()} Dataset "
              f"{dataset_object.target_variable.title()} Distribution")
    plt.xlabel(f"{dataset_object.target_variable.title()}")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{dataset_object.name}_target_distribution.png")
    plt.clf()
    logging.info(f"Saved output class results to {result_path}")

    # create tsne plot
    modeling.dim_reduction_plot(dataset_object)
    logging.info(f"Saved dimension reduction results to {result_path}")

    # move back into original path
    os.chdir(original_path)


# SAMPLE RUN: python main.py --data wine --model dt --tune initial
# main functions
if __name__ == "__main__":

    # get config
    terminal_arg = config_parser()
    model_config = yaml_config_reader(terminal_arg)

    # get dataset
    dataset = set_dataset(terminal_arg.data)
    logging.info(f"Dataset loaded: {terminal_arg.data.capitalize()}")
    logging.info(f"Trained on: {dataset.X_train.columns}")
    dataset.name = model_config['data']

    # for all models, create list of model names
    if terminal_arg.model == 'all':
        model_list = [
            'dt', 'nn', 'boost', 'svm', 'knn'
                      ]
    else:
        model_list = [terminal_arg.model]

    # run specific tuning method
    if terminal_arg.tune == 'initial':
        for model in model_list:
            run_tuning(model,
                       model_config,
                       dataset)

    elif terminal_arg.tune == 'gridsearch':
        for model in model_list:
            run_gridsearch(model,
                           model_config,
                           dataset)

    elif terminal_arg.tune == 'final':
        for model in model_list:
            run_final_results(model,
                              model_config,
                              dataset)

    elif terminal_arg.tune == 'preliminary':
        run_preliminary_analysis(dataset)


