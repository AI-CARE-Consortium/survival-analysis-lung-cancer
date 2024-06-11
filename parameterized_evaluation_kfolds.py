from sklearn.ensemble import RandomForestClassifier
import torch
from sksurv.metrics import as_concordance_index_ipcw_scorer
from pathlib import Path

import evaluation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from data_loading import import_vonko
from data_preprocessing import (calculate_survival_time,
                                encode_selected_variables,
                                imputation)
from evaluation import PartialLogLikelihood, PartialMSE
# import wandb_training
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sklearn.inspection import permutation_importance
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv
from training_survival_analysis import train_model
from data_preprocessing import tumorDataset
from models import TabNetSurvivalRegressor
import argparse
import logging
import explainability
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--imputation_method", type=str, default="none", help="Which imputation method to use")
    parser.add_argument("--model", type=str, default="rsf", help="Which model to use")
    parser.add_argument("--deep_surv_model", type=str, default="none", help="If model is deep_surv, which specific deep_surv model to use")
    parser.add_argument("--tnm", action="store_true", help="Use TNM instead of UICC")
    parser.add_argument("--one-hot", action="store_true", help="Use one-hot encoding instead of label encoding")
    parser.add_argument("--loss", type=str, default="pll", help="Which loss function to use for Tabnet and Deep_Surv")
    parser.add_argument("--imputation_before", action="store_true", help="Impute the data before splitting or afterwards")
    args = parser.parse_args()
    

    model = args.model
    if model not in ["rsf", "cox", "deep_surv", "tabnet"]:
        raise ValueError("Model not implemented")
    if args.imputation_method not in ["none", "KNNImputer", "SimpleImputer", "MissForest"]:
        raise ValueError("Imputation method not implemented")
    if args.loss == "pll":
        loss_fn = PartialLogLikelihood
    elif args.loss == "mse":
        loss_fn = PartialMSE
    else:
        raise ValueError("Loss function not implemented")
    
    
    imputation_features = ["geschl", "alter", "uicc", "histo_gr", "vit_status", "survival_time"]
    selected_features = ["geschl", "alter", "histo_gr", "uicc"]  
    subset = "uicc"
    if args.tnm:
        imputation_features = ["geschl", "alter", "tnm_t", "tnm_n", "tnm_m", "histo_gr", "vit_status", "survival_time"]
        selected_features = ["geschl", "alter", "histo_gr", "tnm_t", "tnm_n", "tnm_m"]
        subset = "tnm"

    config = yaml.safe_load(Path("./config.yaml").read_text())
    base_path = config["base_path"]

    study_name=f"{subset}/missings_imputed_with_{args.imputation_method}"
    if args.one_hot:
        study_name = study_name + "_onehot"
    #If folder does not exist, create it
    if model == "deep_surv":
        dsmodel = args.deep_surv_model
        if dsmodel not in ["minimalistic_network"]:
            raise ValueError("DeepSurv model not implemented")
    path = Path(f"{base_path}/results/{study_name}")
    path.mkdir(parents=True, exist_ok=True)
    #Set up logging
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(f"{path}/evaluation_log_{model}.txt"))
    logger.setLevel(logging.INFO)
    
    

    random_state = 42
    np.random.seed(random_state)



    # Import Dataset + Preprocessing
    vonko = import_vonko(f"{base_path}/data/", oncotree_data=False,
                        processed_data=True, extra_features=True, simplify=True)

    X = vonko["Tumoren"].copy()

    X = calculate_survival_time(X)
    X, encoder = encode_selected_variables(X, imputation_features, na_sentinel=True)
    # X = X.replace(-1, np.nan, inplace=False)
    # X = X.dropna(axis=0, how="any", subset=selected_features)

    y = pd.DataFrame({'vit_status': X['vit_status'].astype(bool),
                    'survival_time': X['survival_time']})
    y = Surv.from_dataframe("vit_status", "survival_time", y)

    # Impute missing values
    
    if args.imputation_method == "none":
        # for each column in selected_features, replace -1 with number of categories + 1
        for feature in selected_features:
            X[feature].replace(-1, len(X[feature].unique())-1, inplace=True)

    if args.imputation_before:
        X = imputation(X, imputation_features=imputation_features, selected_features=selected_features, imputation_method=args.imputation_method, one_hot=args.one_hot,
                       logger=logger, random_state=random_state)

       

    # Select Features for training
    
    # Split between Test and Training for Hyperparameter Tuning
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)


    # create k folds and save them
    kfold = KFold(n_splits=5, shuffle=True, random_state=random_state)
    folds = kfold.split(X_train, y_train)
    model_path = f"{base_path}/results/{study_name}/parameters_study_{model}"
    if model=="tabnet" or model == "deep_surv":
        model_path += f"_{args.loss}"
    best_params = yaml.safe_load(Path(f"{model_path}.yaml").read_text())

    if model == "deep_surv":
        stable_params = {
            "device": config["device"],
            "model": dsmodel,
            "epochs": 300,
            "input_dim": len(X.columns),
            "loss_fn" : loss_fn
        }

        
    logger.info("RESULTS")
    logger.info(f"Best params: {best_params}")
    logger.info("Evaluating best model on test set")
    for i, (train_fold , val_fold) in enumerate(kfold.split(X_train, y_train)):
        if not args.imputation_before:
            X_train_fold = imputation(X_train.iloc[train_fold], imputation_features=imputation_features, selected_features=selected_features, 
                                      imputation_method=args.imputation_method, one_hot=args.one_hot, logger=logger, random_state=random_state)
            X_val_fold = imputation(X_train.iloc[val_fold], imputation_features=imputation_features, selected_features=selected_features,
                                    imputation_method=args.imputation_method, one_hot=args.one_hot, logger=logger, random_state=random_state)
        else:
            X_train_fold = X_train.iloc[train_fold]
            X_val_fold = X_train.iloc[val_fold]
        logger.info(f"Fold {i}:")
        if model == "rsf":
            best_model = RandomSurvivalForest(**best_params, random_state=random_state, n_jobs=32)
            best_model.fit(X_train_fold, y_train[train_fold])
            scores = evaluation.evaluate_survival_model(best_model, X_val_fold, y_train[train_fold],
                                                    y_train[val_fold])
        elif model == "cox":
            best_model = CoxPHSurvivalAnalysis(**best_params)
            best_model.fit(X_train_fold, y_train[train_fold])
            logger.info("Betas:")
            logger.info(best_model.coef_)
            scores = evaluation.evaluate_survival_model(best_model, X_val_fold, y_train[train_fold],
                                                    y_train[val_fold])
        elif model == "deep_surv":
            dataset_train = tumorDataset(X_train_fold, y_train["vit_status"][train_fold], y_train["survival_time"][train_fold])
            best_model, losses, test_eval = train_model(dataset_train, {**stable_params, **best_params})
            best_model.eval()
            y_pred = best_model(torch.Tensor(X_val_fold.values).to(stable_params["device"])).detach().cpu().numpy()

            scores = evaluation.evaluate_survival_model(best_model, X_val_fold.values, y_train[train_fold],
                                                        y_train[val_fold])
        elif model == "tabnet":
            optimizer_params = {
                "lr": best_params["lr"],
                "weight_decay": best_params["weight_decay"]
            }
            best_params["optimizer_params"] = optimizer_params
            best_params_subset = best_params.copy()
            best_params_subset.pop("lr")
            best_params_subset.pop("weight_decay")
            cat_dims = [len(pd.unique(X_train[feature])) for feature in selected_features if feature != "alter"]

            cat_idxs = [0,2,3]
            if subset == "tnm":
                cat_idxs = [0,2,3,4,5]
            best_model = TabNetSurvivalRegressor(cat_idxs=cat_idxs, cat_dims=cat_dims, seed=random_state,
                                         device_name=config["device"], n_a=best_params["n_d"], **best_params_subset)
            y_train_numpy = np.stack((y_train["vit_status"][train_fold], y_train["survival_time"][train_fold]), axis=-1) # np.expand_dims(y_train["vit_status"][train_fold],1) 
            best_model.fit(
                X_train_fold.values, y_train_numpy,
                loss_fn=loss_fn
            )
            scores = evaluation.evaluate_survival_model(best_model, X_val_fold.values, y_train[train_fold],
                                                        y_train[val_fold])
        if True:
            result = permutation_importance(
                best_model, X_val_fold, y_train[val_fold], n_repeats=15, random_state=random_state,
            )
            permutation_importances = pd.DataFrame(
                {k: result[k] for k in ("importances_mean", "importances_std",)},
                index=X_test.columns).sort_values(by="importances_mean", ascending=False)
            logger.info(f"Permutation Importances:")
            logger.info(permutation_importances)
        
        logger.info(scores)

        
        if model == "tabnet":
            logger.info("Tabnet Feature Importance:")
            logger.info(best_model.feature_importances_)
            if i == 0:

                explain_matrix, masks = best_model.explain(X_val_fold.values)
                print(len(masks))
                fig, axs = plt.subplots(1, best_params["n_steps"], figsize=(20,20))

                for j in range(best_params["n_steps"]):
                    axs[j].imshow(masks[j][:50])
                    axs[j].set_title(f"mask {j}")

                plt.savefig(f"{path}/matrix_{i}.png")
                plt.clf()

            shaps = explainability.SHAP(best_model, X_val_fold[::5].values, feature_names=selected_features)

                
        else:
            shaps = explainability.SHAP(best_model, X_val_fold[::5], feature_names=selected_features)
        violin = shaps.plot_violin()
        plt.savefig(f"{path}/violin_fold_{i}_{model}.png")
        plt.clf()
        beeswarm = shaps.plot_beeswarm()
        plt.savefig(f"{path}/beeswarm_fold_{i}_{model}.png")
        plt.clf()
