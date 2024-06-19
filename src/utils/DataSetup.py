# import packages
import logging
import pandas as pd

# import specific modules
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)


class DataSetup:
    def __init__(self,
                 dataset,
                 target_variable):
        # store parameters inputted in class
        self.dataset = dataset
        self.target_variable = target_variable

        # extract features
        self.columns = self.dataset.columns.tolist()
        self.all_features = [col for col in self.dataset.columns.tolist()
                             if col != self.target_variable]
        self.cat_features = self.dataset.columns[self.dataset.dtypes == 'object'].tolist()
        self.num_features = self.dataset.columns[self.dataset.dtypes != 'object'].tolist()

        # make sure to remove target in cat and num features!!
        self.cat_features = [col for col in self.cat_features if col != self.target_variable]
        self.num_features = [col for col in self.num_features if col != self.target_variable]

        # initialize onehot encoded features
        self.dataset_onehot = self.dataset.copy()
        self.dataset_onehot = pd.get_dummies(self.dataset_onehot,
                                             columns=self.cat_features,
                                             prefix=self.cat_features,
                                             drop_first=True)
        self.onehot_features = self.dataset_onehot.columns[self.dataset_onehot.dtypes == 'bool'].tolist()

        # store X (default to all) and y to use for splits
        self.X = self.dataset[self.all_features]
        self.y = self.dataset[self.target_variable]

        # initialize splits
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

        # optional, choose to store variable name
        self.name = None

    @staticmethod
    # format value counts for value + proportion
    def format_value_counts(y_vals,
                            sort_index=True):
        """
        This function is used to format the value counts of a variable into
        'Value (proportion%)' format.

        :param value_counts: the value counts
        :param sort_index: whether to sort the index
        :return: the formatted value counts
        """
        # vary output based on sort index
        if sort_index:
            # get absolute counts
            val_count = pd.concat(
                [y_vals.value_counts(normalize=False).sort_index(),
                 y_vals.value_counts(normalize=True).sort_index()],
                axis=1)
        else:
            val_count = pd.concat(
                [y_vals.value_counts(normalize=False),
                 y_vals.value_counts(normalize=True)],
                axis=1)
        # format value counts
        formatted = (val_count['count'].astype(str) +
                     val_count['proportion'].apply(
                         lambda x: f" ({x:.2%})"))

        return formatted

    # dataset describer
    def describe_dataset(self,
                         sort_index=True):
        """
        This function is used to describe the dataset for an overview of fields,
        variables, target variable, etc.

        :param sort_index: boolean whether to sort the index of the target
                           variable
        """
        # print dataset information
        print(f"Datapoints: {len(self.dataset)}")
        print(
            f"Features: {', '.join(self.all_features)} ({len(self.all_features)} attributes)")
        print(f"Missing Values: {self.dataset.isnull().sum().sum()}")
        print('--------------Target Counts--------------')
        val_count = self.format_value_counts(self.dataset[self.target_variable],
                                             sort_index=sort_index)
        print(f"Target Variable: {val_count}")
        print('-----------------------------------------')

        # print(f"Dataset Null Values: ", dataset.isnull().sum())

    # split data into train and test set
    def save_train_test_split(self,
                              test_split=0.2,
                              random_state=2024,
                              store_splits=True,
                              onehot=False,
                              features=None,
                              smote=False):
        """
        This function is used to create a train test split of the dataset.

        :param test_split: the size of the test set
        :param random_state: the random state
        :param store_splits: whether to store the splits in the class or return
        :param: onehot: if dataset to use to train is the onehot or default version
        :param: features: features used to train and save in self.X
        :param: smote: whether to use SMOTE to balance the dataset
        :return: the train test split
        """
        # if onehot is true, re-save the onehot values to X
        if onehot:
            self.X = self.dataset_onehot

        # slice specific features if not None
        if features is not None:
            self.X = self.X[features]

        # create train test split for normal dataset
        (X_train,
         X_test,
         y_train,
         y_test) = train_test_split(self.X,
                                    self.y,
                                    test_size=test_split,
                                    random_state=random_state)

        # logging message
        # logging.info(f"Splitting data set with test ({test_split}) "
        #              f"(Random State: {random_state})")

        # store splits in class
        if store_splits:
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test
            # logging.info("Splits saved in class object")
        else:
            # logging.info("Splits returned as tuple")
            # return splits
            return X_train, X_test, y_train, y_test

        # use SMOTE if necessary
        if smote:
            # transform the dataset
            oversample = SMOTE(random_state=random_state,
                               n_jobs=-1,
                               k_neighbors=4)
            self.X_train, self.y_train = oversample.fit_resample(self.X_train,
                                                                 self.y_train)
            print("SMOTE applied to dataset")

