# %%
# Import
import sys

sys.path.append('../')
from datenimport_aicare.data_loading import import_vonko
from datenimport_aicare.data_preprocessing import calculate_survival_time, impute, encode_selected_variables
import matplotlib.pyplot as plt
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv
import yaml
from pathlib import Path
import pandas as pd


# %%
#Load data
config = yaml.safe_load(Path("./config.yaml").read_text())
base_path = config["base_path"]
vonko = import_vonko(path=f"{base_path}/aicare/raw/", oncotree_data=False,
                            processed_data=True, extra_features=False, simplify=True)
        
X = vonko["Tumoren"].copy()
X["survival_time"] = calculate_survival_time(X, "vitdat", "diagdat")
imputation_features = ["geschl", "alter", "tnm_t", "tnm_n", "tnm_m", "uicc", "histo_gr", "vit_status", "survival_time"]
            
#X, encoder = encode_selected_variables(X, imputation_features, na_sentinel=True)
# X = X.replace(-1, np.nan, inplace=False)
# X = X.dropna(axis=0, how="any", subset=selected_features)

y = pd.DataFrame({'vit_status': X['vit_status'].astype(bool),
                'survival_time': X['survival_time']})
y = Surv.from_dataframe("vit_status", "survival_time", y)
# %%
#Plot Kaplan-Meier curve
def compare_categories(column: str):
    #print(encoder.encodeTable[column].values)
    #print(encoder.encodeTable[column].codes)
    categories = list(X[column].cat.categories)
    categories.append("nan")
    print(categories)
    for category in categories:#zip(encoder.encodeTable[column].values, encoder.encodeTable[column].codes):
        print(category)#, code)
        mask = X[column] == category
        if str(category) == "nan":
            mask = X[column].isna()
        if not mask.any():
            continue
       
        time, survival_prob, conf_int = kaplan_meier_estimator(
            y["vit_status"][mask],
            y["survival_time"][mask],
            conf_type="log-log")
        #plt.step(time, survival_prob, where="post")
        plt.fill_between(time, conf_int[0], conf_int[1], alpha=0.25, step="post")
        plt.step(time, survival_prob, where="post",
                 label=(column + " = %s") % category)

    plt.ylabel("est. probability of survival $\hat{S}(t)$")
    plt.xlabel("time $t$")
    plt.legend(loc="best")
    

# %%
column = "tnm_t"
compare_categories(column)

plt.savefig(f"{base_path}/results/{column}.png", dpi=500)

plt.show()
# %%
