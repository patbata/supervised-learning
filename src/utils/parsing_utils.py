# import parsing and os packages
import argparse

# import data packages
import pandas as pd
import yaml

# import self helper functions
from .DataSetup import DataSetup


# define arg parser and config reader
def config_parser():
    """
    Parses arguments for main.py
    """
    # parse arguments
    parser = argparse.ArgumentParser()

    # add datatype argument
    parser.add_argument(
        "-d", "--data", type=str, default='wine', help='Dataset to use'
    )

    # add model used to train
    parser.add_argument(
        "-m", "--model", type=str, default='dt', help='Model to use'
    )

    # add parameter for gridsearch vs inital graphs
    parser.add_argument(
        "-t", "--tune", type=str, default='initial', help='Tuning method (initial or gridsearch)'
    )

    return parser.parse_args()


# read config file for model training
def yaml_config_reader(terminal_arg):
    """
    Reads model config file based on data.

    :param terminal_arg: terminal arguments
    """
    # read config file based on data
    with open(f'configs/{terminal_arg.data}.yaml', "r") as f:
        model_config = yaml.safe_load(f)

    return model_config


# set up dataset object
def set_dataset(dataset_name):
    """
    Initializes, fixes and saves dataset object for model running.

    :param dataset_name: name of dataset to load
    """
    filepath = f'data/{dataset_name}-data/{dataset_name}.data'

    # if student, change to csv
    if dataset_name == 'student':
        filepath = f'data/{dataset_name}-data/{dataset_name}.csv'
        dataset = pd.read_csv(filepath, sep=';')
    else:
        dataset = pd.read_csv(filepath)
    # initialize target
    target = None # initialize variable

    # clean wine dataset
    if dataset_name=='wine':
        dataset.columns = ['class', 'alcohol', 'malic_acid', 'ash',
                             'alcalinity_of_ash', 'magnesium',
                             'total_phenols', 'flavanoids',
                             'nonflavanoid_phenols', 'proanthocyanins',
                             'color_intensity', 'hue',
                             'OD280/OD315_of_diluted_wines', 'proline']
        target = 'class'

    # clean poker-hand dataset
    elif dataset_name=='poker-hand':
        dataset.columns = ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4',
                              'S5', 'C5', 'hand']

        # save numerical version of data ("one-hot" encoding)
        onehot_ver = dataset.copy()

        # dictionaries for cleaning
        poker_suits = {1: 'Hearts', 2: 'Spades', 3: 'Diamonds', 4: 'Clubs'}
        poker_hands = {0: 'Nothing in hand', 1: 'One pair', 2: 'Two pairs',
                       3: 'Three of a kind',
                       4: 'Straight', 5: 'Flush', 6: 'Full house',
                       7: 'Four of a kind',
                       8: 'Straight flush', 9: 'Royal flush'}
        # clean suit columns
        for i in range(1, 6):
            dataset['S' + str(i)] = dataset['S' + str(i)].map(poker_suits)

        # clean hand categories
        dataset.hand = dataset.hand.map(poker_hands)

        # set target
        target = 'hand'

    # clean student dataset
    elif dataset_name == 'student':
        # remove classes with only 1 datapoint
        dataset = dataset[~dataset.G3.isin([4, 20,
                                            5, 17, 19,
                                            ])]
        # set target
        target = 'G3'
    # elif dataset_name=='heart'

    # initialize dataset object
    dataset_object = DataSetup(dataset, target)

    # initialize train and test split
    if dataset_name in ['poker-hand']:
        # if dataset has categorical features
        dataset_object.save_train_test_split(
            test_split=0.2,
            random_state=2024,
            store_splits=True,
            onehot=True)
    elif dataset_name == 'student':
        training_features = [
            'school_MS',
            'sex_M',
            'address_U', 'famsize_LE3', 'Pstatus_T',
            # 'Mjob_health', 'Mjob_other', 'Mjob_services',
            'Mjob_teacher',
            # 'Fjob_health', 'Fjob_other', 'Fjob_services',
            'Fjob_teacher',
            # 'reason_home', 'reason_other', 'reason_reputation',
            # 'guardian_mother', 'guardian_other',
            # 'schoolsup_yes', 'famsup_yes',
            # 'paid_yes', 'activities_yes',
            # 'nursery_yes', 'higher_yes',
            # 'internet_yes', 'romantic_yes',
            'age',
            'Medu', 'Fedu',
            'traveltime', 'studytime',
            'failures',
            'famrel', 'freetime', 'goout', 'Dalc', 'Walc',
            'health', 'absences',
            # 'G1', 'G2',
        ]
        dataset_object.save_train_test_split(
            test_split=0.2,
            random_state=2024,
            store_splits=True,
            onehot=True,
            features=training_features,
            smote=False)

    else:
        dataset_object.save_train_test_split(
            test_split=0.2,
            random_state=2024,
            store_splits=True)

    # store dataset name
    dataset_object.dataset_name = dataset_name

    return dataset_object

