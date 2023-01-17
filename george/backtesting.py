import pandas as pd
import numpy as np
from scipy import stats


def log_lik(p: float, obs: pd.Series):
    """
    Likelihood ratio = -2*log(L_1/L_2)
    This function will calculate log(L_i) for any series of binomial obesrvations

    :param p: probability of success
    :param obs: Series of observations, values are 1 or 0
    """
    return obs.apply(lambda x: np.log(((1-p)**(1-x))*(p**x))).sum()


def lr_uc(alpha: float, exceeds: pd.Series):
    """
    Calculate unconditional likelihood ratio statistic for any series of binomial obesrvations

    :param alpha: confidence level between 0 and 1
    :param exceeds: Series of booleans/ints, indicating if the event happened
    """
    # calc pi_hat MLE
    obs = exceeds.astype(int)
    pi_hat = np.mean(obs)

    # return LR_UC for pi_hat and 1-alpha
    return -2*(log_lik(1-alpha, obs)-log_lik(pi_hat, obs))


def p_chi2(x: float, dof: int = 1):
    """
    Calculate the p value for a chi2 distrubution

    :param x: test statistic value
    :param dof: Degrees of freedom for chi2 ist
    """
    return 1-stats.chi2.cdf(x, dof)

