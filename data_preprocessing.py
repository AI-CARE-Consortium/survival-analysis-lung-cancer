import sys
import pandas as pd
import sklearn.impute as impute
import torch
import torch.utils.data
from data_loading import import_vonko
from typing import List, Tuple
import numpy as np
import sklearn
import sklearn.preprocessing as preprocessing

sys.path.append('./')

# sklearn.set_config(transform_output="pandas")


class Encoder():
    '''Encodes categorical variables to numerical variables'''
    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.encodeTable = {}
        self.dataframe = dataframe

    def label_encoding(self, col_name: str, na_sentinel: bool = True) -> pd.DataFrame:

        if pd.api.types.is_categorical_dtype(self.dataframe[col_name]):
            self.dataframe[col_name], self.encodeTable[col_name] = pd.factorize(self.dataframe[col_name],
                                                                                sort=True,
                                                                                use_na_sentinel=na_sentinel)
            # self.dataframe[col_name] = self.dataframe[col_name].cat.codes
        
        return self.dataframe

    def one_hot_encoding(self, col_name: str) -> pd.DataFrame:
        if pd.api.types.is_categorical_dtype(self.dataframe[col_name]):
            one_hot_encoding = pd.get_dummies(self.dataframe[col_name], prefix=col_name)
            self.dataframe = pd.concat([self.dataframe.drop(columns=[col_name]), one_hot_encoding], axis=1)
        return self.dataframe
    


def encode_selected_variables(dataframe: pd.DataFrame, selected_variables: List[str], na_sentinel=True) -> Tuple[pd.DataFrame, Encoder]:
    '''Converts selected categorical variables to numerical variables'''
    encoder = Encoder(dataframe=dataframe)
    for feature in selected_variables:
        dataframe = encoder.label_encoding(feature, na_sentinel=na_sentinel)
    return dataframe, encoder









def calculate_survival_time(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the survival time as time between diagnosis and last vital state
    """
    survival_time = dataframe.loc[:, "vitdat"] - dataframe.loc[:, "diagdat"]
    dataframe["survival_time"] = survival_time.dt.components.days.values
    return dataframe


def calculate_outcome_in_X_years(dataframe: pd.DataFrame, years: int) -> pd.DataFrame:
    """
    Calculate long time survival
    """
    dead = (dataframe["survival_time"] < 365 * years) & (dataframe["vit_status"] == 1)
    dataframe["DeadInXYears"] = dead.values
    alive = dataframe["survival_time"] >= 365 * years
    
    dataframe = dataframe[dead | alive]
    
    return dataframe


class tumorDataset(torch.utils.data.Dataset):
    """
    Torch Dataset Wrapper.
    """
    def __init__(self, selected_features: pd.DataFrame, target: pd.DataFrame, events: pd.DataFrame = None) -> None:
        super().__init__()
        selected_features = preprocessing.MinMaxScaler().fit_transform(selected_features)
        self.selected_features = torch.tensor(selected_features, dtype=torch.float32)
        self.target = torch.tensor(np.asarray(target), dtype=torch.float32)
        if events is not None:
            self.events = torch.tensor(np.asarray(events), dtype=torch.int16)
        else:
            # If no events are given, we assume that all patients are uncensored.
            self.events = torch.ones(len(self.target), dtype=torch.int16)

    def __len__(self) -> int:
        return len(self.selected_features)

    def __getitem__(self, index: int) -> Tuple[torch.tensor, torch.tensor]:
        return self.selected_features[index], self.target[index], self.events[index]


