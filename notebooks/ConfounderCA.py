from sklearn.decomposition import PCA, FastICA, FactorAnalysis
from sklearn.cross_decomposition import PLSRegression, PLSSVD
from sklearn.linear_model import LinearRegression, Ridge

import statsmodels.api as sm
import numpy as np

# Function to sort confounders based on their impact on X and Y
def sort_cf(X, Y, Z, confidence_level=0.95):
    loss, confs_y, confs_x = [], [], []
    
    # Iterate through Z columns
    for i in range(Z.shape[1]):
        # Controlling on Z for Y
        covariatesY = np.hstack((Z[:, i, None], X.reshape(-1, 1)))
        covY_with_intercept = sm.add_constant(covariatesY)
        model_sm_y = sm.OLS(Y, covY_with_intercept).fit()
        p_value_y = model_sm_y.pvalues[1]
        confs_y.append(model_sm_y.conf_int(alpha=1 - confidence_level))

        # Controlling on Z for X
        covariatesX = np.hstack((Z[:, i, None], Y.reshape(-1, 1)))
        covX_with_intercept = sm.add_constant(covariatesX)
        model_sm_x = sm.OLS(Y, covX_with_intercept).fit()
        p_value_x = model_sm_x.pvalues[1]
        confs_x.append(model_sm_x.conf_int(alpha=1 - confidence_level))

        loss.append(p_value_x + p_value_y)
    
    idx = np.argsort(loss)
    losses = np.array(loss)[idx]
    confs_y = np.array(confs_y)[idx]
    confs_x = np.array(confs_x)[idx]

    return idx, losses, confs_x, confs_y

# Function to estimate causal effect
def causal_effect_estimation(X, Y, U, confidence_level=0.95):
    covariates = np.hstack((X.reshape(-1, 1), U))
    cov_intercept = sm.add_constant(covariates)
    model = sm.OLS(Y, cov_intercept).fit()
    causal_effect = model.params[1]
    conf_int = model.conf_int(alpha=1 - confidence_level)[1]
    uncertainty = conf_int[1] - causal_effect
    return causal_effect, uncertainty

class ConfounderCA:
    def __init__(self, dr_method='ICA', XY_causal=True, YX_causal=True, ncp=10):
        self.dr_method = dr_method
        self.XY_causal = XY_causal
        self.YX_causal = YX_causal
        self.ncp = ncp

    def reduce(self, U, X=None, Y=None):
        if self.dr_method == 'ICA':
            self.ica = FastICA(n_components=self.ncp, max_iter=100000, tol=1e-6, random_state=2)
            U_red = self.ica.fit_transform(U)
        elif self.dr_method == 'PCA':
            self.pca = PCA()
            U_red = self.pca.fit_transform(U)
        elif self.dr_method == 'PLS':
            self.pls = PLSSVD()
            U_red = self.pls.fit_transform(U, np.vstack((X, Y)).T)[0]
        return U_red

    def save_weights(self, idx):
        if self.dr_method == 'ICA':
            self.weights = self.ica.mixing_[:, idx]
        elif self.dr_method == 'PCA':
            self.weights = self.pca.components_[idx]
        elif self.dr_method == 'PLS':
            self.weights = self.pls.x_weights_[:, idx]

    def fit_transform(self, U, X, Y, threshold=0.05):
        U_red = self.reduce(U, X, Y)
        U_red = U_red / U_red.std(axis=0)
        self.idx, self.losses, self.conf_x, self.conf_y = sort_cf(X, Y, U_red)
        confounders = U_red[:, self.idx[self.losses <= threshold]]

        self.save_weights(self.idx)

        return confounders
