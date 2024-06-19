# import packages
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import sklearn as sk

# import specific modules
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# import metrics for validation
from sklearn.model_selection import (
    cross_validate,
    GridSearchCV,
    KFold,
    StratifiedKFold,
    learning_curve,
    validation_curve
)

# set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)

# set up plot parameters
plt.rcParams.update({'font.size': 16})


# create title case for each parameter
def title_case(parameter):
    """
    This function is used to convert a parameter into title case.

    :param parameter: the parameter to convert
    :return: the title cased parameter
    """
    return parameter.replace("_", " ").title()


# functionality in sklearn isn't as robust for plotting
def plot_confusion_matrix(cm,
                          classes,
                          model_name,
                          dataset_object,
                          scaler=None):
    """
    This function is used to plot the confusion matrix.

    :param cm: the confusion matrix
    :param classes: the test set
    :param model_name: the name of the model
    :param dataset_object: the dataset object
    """
    # initialize plot
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap='OrRd')

    # format for scaler
    if scaler is not None and scaler != 'None':
        ax.set_xlabel('Predicted Labels (Scaled)')
        ax.set_ylabel('True Labels (Scaled)')
        model_suffix = f" (scaled with {scaler})"
    elif scaler == 'None':
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        model_suffix = ' (unscaled)'
    else:
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        model_suffix = ''
    # labels, title and ticks
    ax.set_title(f'Confusion Matrix of Best {model_name} on\n'
                 f'{dataset_object.name.title()}{model_suffix} Dataset')
    ax.xaxis.set_ticklabels(set(dataset_object.y_train), rotation=45)
    ax.yaxis.set_ticklabels(set(dataset_object.y_train), rotation=0)

    # save plot
    # plt.tight_layout()
    filename = f"confusion_matrix_{model_name}_{dataset_object.name}.png"
    plt.savefig(filename, bbox_inches='tight')

    # clear plot
    plt.clf()


# dataset metrics
def calculate_model_metrics(dataset_object,
                            model_name,
                            metric_used,
                            scores,
                            filename='final_results.txt',
                            scaler=None):
    """
    This function is used to save the average metrics given a set of models
    run using the cross_validate function.

    :param dataset_object: the dataset object which holds the data splits and name
    :param model_name: the name of the model
    :param metric_used: the metric in the score passed to validation
    :param scores: the scores from the cross_validate function
                   (dictionary containing time, estimators,
                   train and test scores)
    :param filename: the filename to save the results
    :param scaler: whether to use a scaler
    """
    # check to see if scaler is needed
    if scaler is not None and scaler != 'None':
        spec_scaler = {'StandardScaler': StandardScaler(),
                       'MinMaxScaler': MinMaxScaler()}[scaler]
        # initialize scaler
        X_train, y_train = spec_scaler.fit_transform(
            dataset_object.X_train), dataset_object.y_train
        # initialize scaler
        X_test = spec_scaler.transform(dataset_object.X_test)
    else:
        # get test split
        X_test = dataset_object.X_test

    # create dataframe with all predictions
    n_y_pred = [est.predict(X_test) for est in
                scores['estimator']]
    y_pred = stats.mode(np.array(n_y_pred).T, axis=-1)[0]

    # create summary dataframe to store model metrics
    scores.pop('estimator')
    final_score_csv = pd.DataFrame(scores)
    final_score_csv.columns = [col.replace("_score", f"_{metric_used}").replace("test","val")
                               for col in final_score_csv.columns]
    # calculate common metrics for each model
    manual_metrics = ['accuracy', 'precision', 'recall', 'f1',
                      'balanced_accuracy']
    manual_scores = dict(zip(manual_metrics,
                             [] * len(manual_metrics)))
    for metric in manual_metrics:
        if 'accuracy' in metric:
            manual_scores[f'test_{metric}'] = \
                [sk.metrics.__getattribute__(f"{metric}_score")(
                    dataset_object.y_test,
                    pred_array)
                    for pred_array in n_y_pred]
        else:
            manual_scores[f'test_weighted_{metric}'] = \
                [sk.metrics.__getattribute__(f"{metric}_score")(
                    dataset_object.y_test,
                    pred_array,
                    average='weighted')
                    for pred_array in n_y_pred]
    # append test scores to final_score_csv
    final_score_csv = pd.concat([final_score_csv,
                                pd.DataFrame(manual_scores)],
                                axis=1)
    final_score_csv.to_csv(f"final_results_{model_name}_{dataset_object.name}.csv")

    # create confusion matrix
    cm = sk.metrics.confusion_matrix(dataset_object.y_test,
                                     y_pred)
    plot_confusion_matrix(cm,
                          classes=set(dataset_object.y),
                          model_name=model_name,
                          dataset_object=dataset_object,
                          scaler=scaler)

    # summarize results into txt)
    with open(filename, 'a+') as f:
        f.write(final_score_csv.mean(axis=0).to_string(index=True)+'\n')


# create a grid search function
def grid_search_results(model_name,
                        params,
                        dataset_object,
                        k_folds=5,
                        stratified=False,
                        seed=2024,
                        scoring='accuracy',
                        scaler=None):
    """
    This function is used to perform a grid search. and return the best fit 
    based on parameters and model.
    
    :param model_name: the model to use (example: DecisionTreeClassifier())
    :param params: the parameters to use (dictionary)
    :param dataset_object: the dataset object to use which contains the train
    :param k_folds: the number of folds for validation
    :param stratified: whether to use stratified k-folds
    :param seed: the random state
    :param scoring: the scoring metric
    :param scaler: whether to use a scaler
    """
    # check to see if scaler is needed
    if scaler is not None and scaler != 'None':
        spec_scaler = {'StandardScaler': StandardScaler(),
                       'MinMaxScaler': MinMaxScaler()}[scaler]
        # initialize scaler
        X_train, y_train = spec_scaler.fit_transform(dataset_object.X_train), dataset_object.y_train
    else:
        # get train and test split
        X_train, y_train = dataset_object.X_train, dataset_object.y_train

    # initialize cross validation object using KFold
    if stratified:
        cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    else:
        cv = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

    # call model object using dictionary
    clf = call_model_object(model_name=model_name,
                            model_params={'random_state': seed})

    # fix for adaboost
    if 'estimator' in params:
        base_est = {'DecisionTree': DecisionTreeClassifier(max_depth=1),
                    'RandomForest': RandomForestClassifier(max_depth=1),
                    # 'GradientBoost': GradientBoostingClassifier,
                    # 'XGBoost': XGBoostClassifierho
                    }
        params['estimator'] = [base_est[i] for i in params['estimator']]

    # initialize grid search object
    grid_search = GridSearchCV(estimator=clf,
                               param_grid=params,
                               cv=cv,
                               n_jobs=-1,
                               verbose=2,
                               return_train_score=True,
                               scoring=scoring)

    # fit the model using
    grid_search.fit(X_train, y_train)

    # write gridsearch summary results to file
    with open('gridsearch_summary.txt', 'a+') as f:
        f.write('-----------------------------------------------------------\n')
        # f.write(f'{dataset_object.name.upper()} DATASET\n')
        f.write(f"MODEL:      {model_name} ({grid_search.best_score_:.4f})\n")
        f.write('-----------------------------------------------------------\n')
        f.write(f"Best Model:        {grid_search.best_estimator_}\n")
        f.write(f"Best Parameters:   {grid_search.best_params_}\n")
        f.write(f"Best Score:        {grid_search.best_score_:.4f}\n")
        f.write(f"Scoring Metric:    {scoring}\n")
        f.write(f"Trained on:        {(', ').join(dataset_object.X_train.columns)}\n")
    f.close()

    return pd.DataFrame(grid_search.cv_results_)


# create plotting function
def create_line_plots(
        plot_type,
        train_scores,
        test_scores,
        param_name,
        title,
        ylabel,
        x_values=None,
        filename=None):
    """
    Creates a line plot with confidence intervals for the training or validation
    curves.

    :param plot_type: the type of plot to create (validation or test)
    :param train_scores: the training scores
    :param test_scores: the test scores
    :param param_name: the parameter name
    :param title: the title of the plot
    :param ylabel: the y-axis label
    :param x_values: the x-axis values
    :param filename: the filename to save the plot
    """
    # setup dataframes
    df_train = pd.melt(pd.DataFrame(train_scores.T),
                       var_name=param_name,
                       value_name='train_score')
    df_test = pd.melt(pd.DataFrame(test_scores.T),
                      var_name=param_name,
                      value_name='test_score')

    # add x values
    if x_values is not None:
        df_train[param_name] = df_train.loc[:, param_name].map(dict(zip(range(0,len(x_values)),
                                                x_values)))
        df_test[param_name] = df_test.loc[:, param_name].map(dict(zip(range(0, len(x_values)),
                                                 x_values)))

    # create summaries
    train_sum = df_train.groupby(param_name).mean().reset_index()
    test_sum = df_test.groupby(param_name).mean().reset_index()

    # create lineplot with confidence intervals
    sns.lineplot(data=df_train, x=param_name, y='train_score', markers=True,
                 dashes=False, legend=True, label='Train')
    sns.lineplot(data=df_test, x=param_name, y='test_score', markers=True,
                 dashes=False, legend=True, label=plot_type.capitalize())
    sns.scatterplot(data=train_sum, x=param_name, y='train_score')
    sns.scatterplot(data=test_sum, x=param_name, y='test_score')

    # add labels
    plt.legend()
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(title_case(param_name))

    if x_values is not None:
        plt.xticks(x_values, rotation=45)

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    else:
        plt.show()

    # clean plot area
    plt.clf()


# create model call function
def call_model_object(model_name,
                      model_params=None):
    """
    This function is used to call a model object using a dictionary.

    :param model_name: the name of the model
    :param model_params: the parameters of the model
    :return: the model object
    """
    # create dictionary of models
    MODELS = {
        'DecisionTree': DecisionTreeClassifier,
        'NeuralNetwork': MLPClassifier,
        'BoostedDT': AdaBoostClassifier,
        'SVM': SVC,
        'KNN': KNeighborsClassifier,
    }

    # call model object using dictionary
    if model_params == 'None' or model_params is None:
        model_params = {}

    # remove seed in model params
    if model_name == 'KNN':
        model_params.pop('random_state', None)

    # initialize model object
    clf = MODELS[model_name](**model_params)

    return clf


# create learning curve with model input
def plot_learning_curve(model_name,
                        dataset_object,
                        k_folds=5,
                        seed=2024,
                        scoring='accuracy',
                        model_params=None,
                        stratified=False,
                        scaler=None):
    """
    This function is used to plot the learning curve of a model. Model passed
    into the function should be initialized. This function shows if the model
    over or underfits the data.

    :param model: the model to use
    :param dataset_object: the dataset object to use which contains the train
    and test split stored in it
    :param k_folds: the number of folds for validation
    :param seed: the random state
    :param scoring: the scoring metric
    :param model_params: the parameters of the model. If set to None, uses the
    default of the object
    :param stratified: whether to use stratified k-folds
    :param scaler: whether to use a scaler
    """
    # check to see if scaler is needed
    if scaler is not None and scaler != 'None':
        spec_scaler = {'StandardScaler': StandardScaler(),
                       'MinMaxScaler': MinMaxScaler()}[scaler]
        # initialize scaler
        X_train, y_train = spec_scaler.fit_transform(
            dataset_object.X_train), dataset_object.y_train
    else:
        # get train and test split
        X_train, y_train = dataset_object.X_train, dataset_object.y_train

    # initialize cross validation object using KFold
    if stratified:
        cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    else:
        cv = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

    # make sure to evaluate model params
    if 'hidden_layer_sizes' in model_params:
        model_params['hidden_layer_sizes'] = eval(model_params['hidden_layer_sizes'])
        # fix for adaboost
    elif 'estimator' in model_params:
        base_est = {'DecisionTree': DecisionTreeClassifier,
                    'RandomForest': RandomForestClassifier,
                    'GradientBoost': GradientBoostingClassifier}
        model_params['estimator'] = base_est[model_params['estimator']]()

    # initialize model object using function call
    model = call_model_object(model_name, model_params)

    # create learning curves (train_sizes are used to split to validation?)
    train_percent = np.linspace(0.1, 1.0, 10)
    (train_sizes,
     train_scores,
     test_scores) = learning_curve(model,
                                   X_train,
                                   y_train,
                                   cv=cv,
                                   n_jobs=-1,
                                   train_sizes=train_percent,
                                   scoring=scoring)

    # plot learning curve
    create_line_plots(
        plot_type='validation',
        train_scores=train_scores,
        test_scores=test_scores,
        param_name='train_size',
        title=f"Learning Curve for\n{model_name} (kfolds={k_folds})",
        ylabel=title_case(scoring),
        x_values=train_percent,
        filename=f"learning_{model_name}.png"
    )


# create validation curve that plots hyperparameters given a model
def plot_validation_curve(model_name,
                          dataset_object,
                          param_name,
                          param_range,
                          k_folds=5,
                          seed=2024,
                          scoring='accuracy',
                          model_params=None,
                          stratified=False,
                          scaler=None):
    """
    Given a certain hyperparameter and its range, this function plots the
    validation curves

    :param model_name: the model to use
    :param dataset_object: the dataset object to use which contains the train
    and test split stored in it
    :param param_name: the parameter to be tested
    :param param_range: the range of the parameter
    :param k_folds: the number of folds for validation
    :param seed: the random state
    :param scoring: the scoring metric
    :param model_params: the parameters of the model. If set to None, uses the
    default of the object
    :param stratified: whether to use stratified k-folds
    :param scaler: whether to use a scaler
    """
    # check to see if scaler is needed
    if scaler is not None and scaler != 'None':
        spec_scaler = {'StandardScaler': StandardScaler(),
                       'MinMaxScaler': MinMaxScaler()}[scaler]
        # initialize scaler
        X_train, y_train = spec_scaler.fit_transform(
            dataset_object.X_train), dataset_object.y_train
    else:
        # get train and test split
        X_train, y_train = dataset_object.X_train, dataset_object.y_train

    # call model object using dictionary
    clf = call_model_object(model_name, model_params)

    # initialize cross validation object using KFold
    if stratified:
        cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    else:
        cv = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

    # special case for layers
    x_values = param_range
    param_model = param_name
    if param_name == 'depth_hidden_layer':
        param_model = 'hidden_layer_sizes'
        param_range = [tuple([100]*i) for i in param_range]
    elif param_name == 'estimator':
        base_est = {'DecisionTree': DecisionTreeClassifier,
                    'RandomForest': RandomForestClassifier,
                    'GradientBoost': GradientBoostingClassifier}
        param_range = [base_est[i]() for i in
                       param_range]

    # create validation curve
    train_scores, val_scores = validation_curve(clf,
                                                X_train,
                                                y_train,
                                                param_name=param_model,
                                                param_range=param_range,
                                                cv=cv,
                                                scoring=scoring,
                                                n_jobs=-1)

    # plot validation curve
    create_line_plots(
        plot_type='validation',
        train_scores=train_scores,
        test_scores=val_scores,
        param_name=param_name,
        title=f"Validation Curve for {model_name} with\n"
              f"{title_case(param_name)} (kfolds={k_folds})",
        ylabel=title_case(scoring),
        x_values=x_values,
        filename=f"valcurve_{model_name}_{param_name}.png"
    )


# get elapsed time and model metrics of "best model"
def get_best_model(model_name,
                   model_params,
                   dataset_object,
                   k_folds=5,
                   stratified=False,
                   seed=2024,
                   scoring='accuracy',
                   scaler=None,
                   filename='final_results.txt'):
    """
    This function is used to get the best model using grid search and return
    the elapsed time and model metrics.

    :param model_name: the model to use
    :param model_params: the parameters to use
    :param dataset_object: the dataset object to use which contains the train
    and test split stored in it
    :param k_folds: the number of folds for validation
    :param stratified: whether to use stratified k-folds
    :param seed: the random state
    :param scoring: the scoring metric
    :param scaler: whether to use a scaler
    """
    # check to see if scaler is needed
    if scaler is not None and scaler != 'None':
        spec_scaler = {'StandardScaler': StandardScaler(),
                       'MinMaxScaler': MinMaxScaler()}[scaler]
        # initialize scaler
        X_train, y_train = spec_scaler.fit_transform(
            dataset_object.X_train), dataset_object.y_train
    else:
        # get train and test split
        X_train, y_train = dataset_object.X_train, dataset_object.y_train

    # write gridsearch summary results to file
    with open(filename, 'a+') as f:
        f.write('-----------------------------------------------------------\n')
        # f.write(f'{dataset_object.name.upper()} DATASET\n')
        f.write(f"MODEL:             {model_name}\n")
        f.write(f"Best Parameters:   {model_params}\n")
    f.close()

    # make sure to evaluate model params
    if 'hidden_layer_sizes' in model_params:
        model_params['hidden_layer_sizes'] = eval(
            model_params['hidden_layer_sizes'])
        # fix for adaboost
    elif 'estimator' in model_params:
        base_est = {'DecisionTree': DecisionTreeClassifier,
                    'RandomForest': RandomForestClassifier,
                    'GradientBoost': GradientBoostingClassifier}
        model_params['estimator'] = base_est[model_params['estimator']]()

    # call the model
    best_clf = call_model_object(model_name, model_params)

    # call cv object
    if stratified:
        cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    else:
        cv = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

    # get scores from scoring in cross validate
    scores = cross_validate(
                        best_clf,
                        X_train,
                        y_train,
                        cv=cv,
                        scoring=scoring,
                        n_jobs=-1,
                        return_train_score=True,
                        return_estimator=True,
    )

    # create a loss curve if nn
    if model_name == 'NeuralNetwork':
        # create loss curve for the 5 folds
        loss_curves = [arr.loss_curve_ for arr in scores['estimator']]

        # setup dataframes
        df_loss = pd.melt(pd.DataFrame(loss_curves),
                          var_name='fold',
                          value_name='loss')

        # create summaries
        loss_avg = df_loss.groupby('fold').mean().reset_index()

        # # create lineplot with confidence intervals
        sns.lineplot(data=df_loss, x='fold', y='loss', markers=True,
                     dashes=False, legend=True,)
        # sns.scatterplot(data=loss_avg, x='fold', y='loss', size=2)

        # add labels
        plt.title(f'Loss Curve of Neural Network\n(Split into {k_folds} Folds)'
                  f' on {dataset_object.name.title()} Dataset')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')

        # save losses
        df_loss.to_csv(f"losses_{dataset_object.name}.csv")

        # save plot then clear
        plt.savefig(f"{dataset_object.name}_nn_loss_curve.png", bbox_inches='tight')
        plt.clf()

    # store metrics (saves in written file)
    calculate_model_metrics(dataset_object,
                            model_name,
                            metric_used=scoring,
                            scores=scores,
                            filename=filename,
                            scaler=scaler)

    return scores


# create function for tsne and pca plot
def dim_reduction_plot(dataset_object):
    """
    This function is used to create a tsne and pca plot of the dataset.

    :param dataset_object: the dataset object to use which contains the train
    and test split stored in it
    """
    # set palette
    sns.set_palette("husl")

    # tsne portion
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(dataset_object.X)

    # PCA portion
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(dataset_object.X)

    # append to dataframe
    df_copy = dataset_object.X.copy()
    df_copy['pca-one'] = pca_result[:, 0]
    df_copy['pca-two'] = pca_result[:, 1]
    df_copy['pca-three'] = pca_result[:, 1]
    df_copy['tsne-2d-one'] = tsne_results[:, 0]
    df_copy['tsne-2d-two'] = tsne_results[:, 1]
    df_copy[dataset_object.target_variable] = dataset_object.y

    # tsne plot
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue=dataset_object.target_variable,
        data=df_copy,
        legend="full",
        palette='Set2'
    )
    plt.title(f"t-SNE Plot of {dataset_object.name.title()} Feature Dataset")
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend(ncol=1)
    plt.savefig(f"{dataset_object.name}_tsne.png", bbox_inches='tight')
    # plt.tight_layout()
    plt.clf()

    # pca plot
    sns.scatterplot(
        x="pca-one", y="pca-two",
        hue=dataset_object.target_variable,
        data=df_copy,
        legend="full",
        palette='Set2'
    )
    plt.title(f"PCA Plot of {dataset_object.name.title()} Feature Dataset")
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.tight_layout()
    plt.savefig(f"{dataset_object.name}_pca.png", bbox_inches='tight')
    plt.clf()

