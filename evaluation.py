from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, brier_score,\
    cumulative_dynamic_auc, integrated_brier_score
import numpy as np
from typing import Dict
import torch
from scipy.stats import norm
from sksurv.nonparametric import kaplan_meier_estimator
from pytorch_tabnet.tab_model import TabNetRegressor


def evaluate_survival_model(model, X_test, y_train_numpy, y_test_numpy) -> Dict:
    """
    Evaluate a survival model on the test set with multiple metrics.
    model: Survival model (can be CoxPH, RSF, Pytorch Model or TabNet)
    X_test: Matrix with the variables used for prediction
    y_train_numpy: Survival times for training data to estimate the censoring distribution from. A structured array containing the binary event indicator as first field, and time of event or time of censoring as second field.
    y_test_numpy: Survival times of test data. A structured array containing the binary event indicator as first field, and time of event or time of censoring as second field.
    """


    times = np.unique(np.quantile(y_train_numpy['survival_time'][y_train_numpy['vit_status'] == 1], np.linspace(0.1, 1, 20)))
    
    if isinstance(model, torch.nn.Module):
        model.eval()
        model.cpu()
        with torch.no_grad():
            y_pred = np.squeeze(model(torch.Tensor(np.array(X_test))).detach().cpu().numpy())
          
    else:
        y_pred = np.squeeze(model.predict(X_test))
        c_index = concordance_index_censored(y_test_numpy['vit_status'], y_test_numpy['survival_time'], y_pred)[0]
   
    auc, mean_auc = cumulative_dynamic_auc(y_train_numpy, y_test_numpy, y_pred, times=times)

    if isinstance(model, torch.nn.Module) or isinstance(model, TabNetRegressor):
        # These models do not return a survival function, so we have to calculate it ourselves
        baseline_hazard_times, baseline_hazard = kaplan_meier_estimator(y_test_numpy["vit_status"], y_test_numpy["survival_time"])
        print(np.mean(y_pred))
        max_value = np.log(np.log(np.finfo(np.float32).max )) 
        clipped_y_pred = np.clip(y_pred, -max_value, max_value)
        rates = np.array([baseline_hazard ** np.exp(clipped_y_pred_i) for clipped_y_pred_i in clipped_y_pred])
        surv_prob = []
        for time in times:
            kaplan_index = np.argmax(baseline_hazard_times > time)
            surv_prob.append(rates[:, kaplan_index])
        surv_prob = np.column_stack(surv_prob)
        ibs = integrated_brier_score(y_train_numpy, y_test_numpy, surv_prob, times) 
    else:
        surv_prob = np.row_stack([
            fn(times)
            for fn in model.predict_survival_function(X_test)
        ])
        ibs = integrated_brier_score(y_train_numpy, y_test_numpy, surv_prob, times)


    return {
        "c_index": c_index,
        "mean_auc": mean_auc,
        "ibs": ibs
    }





def PartialLogLikelihood(logits, fail_indicator, times=None, ties="noties"):
    '''
    Implementation of partial log-likelihood loss function from https://github.com/runopti/stg/blob/master/python/stg/losses.py
    fail_indicator: 1 if the sample fails, 0 if the sample is censored.
    logits: raw output from model 
    ties: 'noties' or 'efron' or 'breslow'
    '''
    logL = 0
    # get time order for prediction and fails
    if times is None:
        times = torch.arange(logits.shape[0]).to(logits.device)
    time_index = torch.argsort(-times)
    logits = logits[time_index]
    fail_indicator = fail_indicator[time_index]
    times = times[time_index]
    if ties == 'noties':
        
        log_risk = torch.logcumsumexp(logits, 0)
        likelihood = logits - log_risk
        # dimension for E: np.array -> [None, 1]
        uncensored_likelihood = likelihood * fail_indicator
        logL = -torch.sum(uncensored_likelihood)
    elif ties == "breslow":
    # calculate the log-likelihood with Breslow approximation
        for t in torch.unique(times):
            ix = (times == t).nonzero().squeeze()
            if ix.numel() > 1:
                # compute the gradient at each event time using Breslow approximation
                h = torch.exp(logits[ix])
                cumsum_hazard_ratio = torch.cumsum(h, dim=0)
                g = torch.sum(cumsum_hazard_ratio) / torch.sum(h)
                logL -= g
            else:
                # compute the log-likelihood for uncensored samples
                logL -= logits[ix] * fail_indicator[ix]

    # negative average log-likelihood
    # observations = torch.sum(fail_indicator, 0)
    return logL
