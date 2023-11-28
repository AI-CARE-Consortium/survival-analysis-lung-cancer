import pandas as pd
import numpy as np
from typing import Dict


def aggregation_function(x: pd.Series) -> pd.Series:
    '''
    Aggregation function for the aggregation of data depending on data type.
    
    Parameters:
        x (pd.Series): Series to aggregate
    '''
    x = x.dropna()
    if len(x) == 0:
        return pd.NA
    if x.dtype == np.float64:
        return x.mean()
    elif x.dtype == pd.StringDtype():
        return ", ".join(x.unique())
    elif x.dtype == pd.Timestamp:
        return x.max()
    return x.iloc[0]


def import_vonko(path: str, oncotree_data: bool = False, processed_data: bool = False, extra_features: bool = False,
                 aggregate_therapies: bool = False, simplify: bool = True) -> Dict:
    '''
    Imports data from csv files and returns a dictionary containing the dataframes.
    
    Parameters:
        path (str): Path to the data folder
        oncotree_data (bool): If True, the morponcotree data is used
        processed_data (bool): If True, the processed data is used
        extra_features (bool): If True, additional features are added to the data
        aggregate_therapies (bool): If True, therapies are aggregated
    '''

    vonko = {}

    # Create path to Tumoren file based on parameters
    tumoren_relative_path = "Tumoren"
    if processed_data:
        tumoren_relative_path += "_aufbereitet"
    if oncotree_data:
        tumoren_relative_path += "_oncotree"
    tumoren_relative_path += ".csv"
    
    # Read data from csv files
    diag_data = pd.read_csv(path + tumoren_relative_path, header=0, sep=";", dtype=pd.StringDtype())
    vonko["MET_PT"] = pd.read_csv(path + "MET_PT.csv", header=0, quotechar='"', sep=";", dtype=pd.StringDtype())
    vonko["MET_VM"] = pd.read_csv(path + "MET_Verlauf.csv", header=0, quotechar='"', sep=";", dtype=pd.StringDtype())
    vonko["OP"] = pd.read_csv(path + "OP.csv", header=0, quotechar='"', sep=";", dtype=pd.StringDtype(), na_values=[""])
    vonko["ST"] = pd.read_csv(path + "ST.csv", header=0, quotechar='"', sep=";", dtype=pd.StringDtype())
    vonko["SYS"] = pd.read_csv(path + "SY.csv", header=0, quotechar='"', sep=";", dtype=pd.StringDtype())
    vonko["VM"] = pd.read_csv(path + "VM.csv", header=0, quotechar='"', sep=";", dtype=pd.StringDtype())

    # Define categorical and numeric columns
    cat_cols = ['geschl', 'diag_sich', 'diag_seite', 'diag_icd', 'topo_icdo',
                'morpho_icdo', 'morpho_icdo_version', 'grading', 'tnm_r',
                'c_tnm_version', 'c_tnm_r', 'c_tnm_t', 'c_tnm_n', 'c_tnm_m', 'c_tnm_l', 'c_tnm_v', 'c_tnm_pn', 'c_uicc',
                'p_tnm_version', 'p_tnm_r', 'p_tnm_t', 'p_tnm_n', 'p_tnm_m', 'p_tnm_l', 'p_tnm_v', 'p_tnm_pn', 'p_uicc',
                'tnm_version', 'tnm_r', 'tnm_t', 'tnm_n', 'tnm_m', 'uicc']
    num_cols = ['alter', 'tod_alter', 'lk_befall', 'lk_unters', 'sentinel_befall', 'sentinel_unters', 'vit_status']

    # Add oncotree column if oncotree conversion has been used
    if oncotree_data:
        cat_cols.append('oncotree')

    # Add processed histology column if processed data has been used
    if processed_data:
        cat_cols.append('histo_gr')
        cat_cols.append("morpho_kurz")

    # Exclude Data without vital status and date

    diag_data = diag_data[~diag_data["vitdat"].isna()]
    diag_data = diag_data[~diag_data["vit_status"].isna()]
    
    # Convert date columns to datetime
    diag_data["vitdat"] = pd.to_datetime(diag_data["vitdat"], format="%m.%Y")
    diag_data["diagdat"] = pd.to_datetime(diag_data["diagdat"], format="%m.%Y")
    # Add 14 days to the vitdat and diagdat
    diag_data["vitdat"] = diag_data["vitdat"] + pd.to_timedelta(14, unit="d")
    diag_data["diagdat"] = diag_data["diagdat"] + pd.to_timedelta(14, unit="d")

    vonko["ST"]["stdat_beginn"] = pd.to_datetime(vonko["ST"]["stdat_beginn"], format="%d.%m.%Y")
    vonko["ST"]["stdat_ende"] = pd.to_datetime(vonko["ST"]["stdat_ende"], format="%d.%m.%Y")
    vonko["SYS"]["sydat_beginn"] = pd.to_datetime(vonko["SYS"]["sydat_beginn"], format="%d.%m.%Y")
    vonko["SYS"]["sydat_ende"] = pd.to_datetime(vonko["SYS"]["sydat_ende"], format="%d.%m.%Y")
    vonko["OP"]["opdat"] = pd.to_datetime(vonko["OP"]["opdat"], format="%d.%m.%Y")
    vonko["VM"]["vmdat"] = pd.to_datetime(vonko["VM"]["vmdat"], format="%d.%m.%Y")

    # Plausibility check
    diag_data = diag_data[diag_data["vitdat"] >= diag_data["diagdat"]]

    # Add extra features which are not in the original data
    if extra_features:

        diag_data_stmerge = diag_data.merge(vonko["ST"], left_on="tunr", right_on="tunr", how="left")
        filtered_st = diag_data_stmerge[(diag_data_stmerge["stdat_beginn"] - diag_data_stmerge["diagdat"]).dt.days < 180]
        diag_data["has_ST"] = diag_data["tunr"].isin(filtered_st["tunr"]).astype(np.int8())

        diag_data_sysmerge = diag_data.merge(vonko["SYS"], left_on="tunr", right_on="tunr", how="left")
        filtered_sys = diag_data_sysmerge[(diag_data_sysmerge["sydat_beginn"] - diag_data_sysmerge["diagdat"]).dt.days
                                          < 180]
        diag_data["has_SYS"] = diag_data["tunr"].isin(filtered_sys["tunr"]).astype(np.int8())

        diag_data_opmerge = diag_data.merge(vonko["OP"], left_on="tunr", right_on="tunr", how="left")
        filtered_op = diag_data_opmerge[(diag_data_opmerge["opdat"] - diag_data_opmerge["diagdat"]).dt.days < 180]
        diag_data["has_OP"] = diag_data["tunr"].isin(filtered_op["tunr"]).astype(np.int8())

        # diag_data['has_ST'] = diag_data['tunr'].isin(vonko['ST']['tunr']).astype(np.int8())
        # diag_data['has_SYS'] = diag_data['tunr'].isin(vonko['SYS']['tunr']).astype(np.int8())
        # diag_data['has_OP'] = diag_data['tunr'].isin(vonko['OP']['tunr']).astype(np.int8())
        cat_cols.extend(['has_ST', 'has_SYS', 'has_OP'])

    # Aggregate data if start date is the same
    if aggregate_therapies:
        vonko["ST"] = vonko["ST"].groupby(["tunr", "stdat_beginn"], as_index=False).agg(lambda x: aggregation_function(x))
        vonko["SYS"] = vonko["SYS"].groupby(["tunr", "sydat_beginn"], as_index=False).agg(lambda x: aggregation_function(x))

    if simplify:
        # Convert categoricals to categorical data type
        # Remove trailing characters from uicc
        
        uicc_subset = diag_data["uicc"].str.split("[^IV]", regex=True).str[0].str.strip()
        uicc_subset = uicc_subset.replace(r'^\s*$', pd.NA, regex=True)
        diag_data.__setitem__("uicc", uicc_subset.values)
        for tnm_col in ["tnm_t", "tnm_n", "tnm_m"]:
            tnm_subset = diag_data[tnm_col].str.split("[^0-4]", regex=True).str[0].str.strip()
            tnm_subset = tnm_subset.replace(r'^\s*$', pd.NA, regex=True)
            diag_data.__setitem__(tnm_col, tnm_subset.values)
        
        def lambda_histo_dichtomy(x):
            if x == '3':
                return 0
            elif x in ['1', '2', '4', '5']:
                return 1
            else:
                return 2
        histo_subset = diag_data["histo_gr"].apply(lambda_histo_dichtomy)
        diag_data.__setitem__("histo_gr", histo_subset.values)

        # exclude patients with sarcoma or unknown/other histology
        diag_data = diag_data[diag_data["histo_gr"] != 2]

    for cat_col in cat_cols:
        if diag_data[cat_col].dtype == pd.StringDtype():
            diag_data[cat_col] = diag_data[cat_col].str.upper().str.strip()
        diag_data[cat_col] = diag_data[cat_col].astype(pd.CategoricalDtype(ordered=True))

    # Convert numerics to numeric data type
    for num_col in num_cols:
        diag_data[num_col] = pd.to_numeric(diag_data[num_col])

    # Exclude patients with age > 90
    # diag_data = diag_data[diag_data["alter"] < 90]

    # Convert zustand to unified format
    karnofsky_dict = dict.fromkeys(['100%', '90%'], '0')
    karnofsky_dict.update(dict.fromkeys(['80%', '70%'], '1'))
    karnofsky_dict.update(dict.fromkeys(['60%', '50%'], '2'))
    karnofsky_dict.update(dict.fromkeys(['40%', '30%'], '3'))
    karnofsky_dict.update(dict.fromkeys(['20%', '10%'], '4'))
    karnofsky_dict.update(dict.fromkeys(['U', pd.NA], np.nan))
    diag_data['zustand'] = [karnofsky_dict.get(zustand, zustand) for zustand in diag_data['zustand']]
    diag_data['zustand'] = diag_data['zustand'].astype(pd.CategoricalDtype(ordered=True))
    # diag_data['zustand'] = diag_data['zustand'].astype(np.int8())
    vonko["Tumoren"] = diag_data

    return vonko


