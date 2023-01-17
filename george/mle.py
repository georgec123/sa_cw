import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import scipy.stats as stats
from scipy.special import hyp2f1, beta
from abc import ABC, abstractmethod

from scipy.integrate import quad
from scipy.optimize import minimize_scalar

import backtesting as bt

from typing import Iterable


class Distribution(ABC):
    def __init__(self) -> None:

        self.sol = None
        self.name: str = None
        self.num_parameters: int = None
        self.parameters: str = None
        self.sample_size: int = None
        self.data: pd.DataFrame = None
        return

    @abstractmethod
    def log_likelihood(x: float, *dist_params: float) -> float:
        """
        Calculate log likelihood of observing x, given a distrubution 
        with these parameters.
        """
        pass

    @abstractmethod
    def pdf(x: float) -> float:
        """
        Return value of probability density function evaluated at x.
        """
        pass

    def cdf(self, x: float) -> float:
        """
        Integrate the pdf function to return Cumulative Density value
        """
        mu = self.data.mean()
        std = self.data.std()

        output = quad(lambda x: self.pdf(x, *self.sol.x), mu-20*std, x)
        return output[0]

    def quantile(self, alpha: float) -> float:
        """
        Return inverse cdf for distrubution.
        Inverse is found by root finding (cdf(x)-alpha)^2
        """
        return minimize_scalar(lambda x: (self.cdf(x)-alpha)**2).x

    def expected_shortfall(self, alpha: float) -> float:
        """
        Calculate expected shortfall by integrating xf(x)
        +inf is replace with mu+20*std for efficiency
        """
        mu = self.data.mean()
        std = self.data.std()
        integral = quad(lambda x: x * self.pdf(x, *self.sol.x),
                        self.quantile(alpha), mu+20*std)

        return integral[0]/(1-alpha)

    def print_params(self):
        """
        Print out distrubution parameter names and values
        """
        for idx, param in enumerate(self.parameters):
            print(f"{param}: {self.sol.x[idx]}")

        return

    def info_dict(self) -> dict:
        """
        Return information dict about the quality of distrubition fit.
        Quality of fit is measured by the following metrics:
        - AIC
        - AICC
        - BIC
        - CAIC
        - HQC
        """

        info_dict = {'dist': self.name,
                     'LL': self.sol.fun, 'AIC': self.aic(),
                     'AICC': self.aicc(), 'BIC': self.bic(),
                     'CAIC': self.caic(), 'HQC': self.hqc(),
                     'pydist': self
                     }

        return info_dict

    def plot_dist(self, ax: plt.Axes = None) -> plt.Axes:
        """
        Plot observed data and density of distrubutio fit
        """
        if ax is None:
            _, ax = plt.subplots(1, 1)

        mu = self.data.mean()
        std = self.data.std()

        linspace = np.linspace(mu-4*std, mu+4*std, 101)

        ax.hist(self.data.values, bins=50, density=True)
        density = self.pdf(linspace, *self.sol.x)

        ax.plot(linspace, density)
        ax.set_title(f"{self}")

        return ax

    def kupic_plot(self, quantiles: Iterable[float], ax: plt.Axes = None) -> plt.Axes:

        # Calculate value at risk for each quantile
        vars = [self.quantile(quantile) for quantile in quantiles]

        p_vals = []

        for alpha, var in zip(quantiles, vars):

            # get p value for each quantile based on
            # observed and expected var exceedances
            viol_mask = (self.data > var)
            likelihood_uc = bt.lr_uc(alpha, viol_mask)
            p_val_uc = bt.p_chi2(likelihood_uc)
            p_vals.append(p_val_uc)

        if ax is None:
            _, ax = plt.subplots(1, 1)

        ax.plot(quantiles, p_vals, color='k')
        ax.axhline(0.05, color='r')

        return ax

    @abstractmethod
    def mle(self, x):
        """
        Fit parameters of the distrubution using maximum likelihood estimation
        """
        self.sample_size = x.shape[0]
        self.data = x

    def __repr__(self) -> str:
        if self.sol is None:
            repr_str = f"{self.name} - Unfit"
        else:
            repr_str = f"{self.name}("

            param_list = []
            for p_name, p in zip(self.parameters, self.sol.x):
                param_list.append(f"{p_name}={p:.3f}")

            repr_str = f"{repr_str}{', '.join(param_list)})"

        return repr_str

    def aic(self) -> float:
        """
        The AIC of an MLE estiamte equals 2*number of parameters - value of loglikelihood
        Inputs:
        k-number of parameters in distribution
        L-the value of the loglikelihood function
        """
        return 2*self.num_parameters + 2*self.sol.fun

    def bic(self) -> float:
        """
        The BIC of an MLE estiamte equals #parameters * log(sample size) - 2 * value of loglikelihood
        Inputs:
        k-number of parameters in distribution
        L-the value of the loglikelihood function
        """
        return self.num_parameters*np.log(self.sample_size) + 2*self.sol.fun

    def caic(self) -> float:
        """
        The BIC of an MLE estiamte equals #parameters * (log(sample size)+1) - 2 * value of loglikelihood
        Inputs:
        k-number of parameters in distribution
        L-the value of the loglikelihood function
        """
        return self.num_parameters*(np.log(self.sample_size)+1) + 2*self.sol.fun

    def aicc(self) -> float:
        """
        Corrected Akaike information criterion
        """
        k = self.num_parameters
        n = self.sample_size

        return self.aic() + 2*k*(k+1)/(n-k-1)

    def hqc(self) -> float:
        """
        Hannan-Quinn criterion
        """
        k = self.num_parameters
        n = self.sample_size

        return 2*self.sol.fun + 2*k*np.log(np.log(n))

    def kol_smir(self) -> float:
        """
        Kolmogorov-Smirnov test
        """
        cdfdata = list(map(self.cdf, self.data))
        return stats.kstest(self.data, cdfdata)

    def kappa(self, nu: float) -> float:
        """
        Kappa function used in density of many child distrubutions
        """
        return 1/(np.sqrt(nu)*beta(nu/2, 1/2))


class Laplace(Distribution):

    def __init__(self) -> None:
        super().__init__()
        self.num_parameters = 2
        self.parameters = ['mu', 'b']
        self.name = 'Laplace'

    @staticmethod
    def pdf(x, mu, b):
        return np.exp(-np.abs(x-mu)/b)/(2*b)

    def log_likelihood(self, x, mu, b):
        return np.sum(np.log(self.pdf(x, mu, b)))

    def mle(self, x):
        super().mle(x)
        def objfun(theta): return -1*self.log_likelihood(x, *theta)
        bnds = ((-np.inf, np.inf), (1e-15, np.inf))  # bounds on parameters

        init_theta = [np.mean(x), 1]

        sol = minimize(objfun, init_theta, method='Nelder-Mead', bounds=bnds)
        self.sol = sol
        
        return sol.x


class StudentT(Distribution):

    def __init__(self) -> None:

        super().__init__()
        self.num_parameters = 3
        self.parameters = ['mu', 'sigma', 'nu']
        self.name = 'Student T'

    def pdf(self, x, mu, sigma, nu):
        return (self.kappa(nu)/sigma) * (1 + ((x-mu)**2)/(nu * sigma**2))**(-(1+nu)/2)

    def log_likelihood(self, x, mu, sigma, nu):
        return np.sum(np.log(self.pdf(x, mu, sigma, nu)))

    def mle(self, x):
        super().mle(x)
        def objfun(theta): return -1*self.log_likelihood(x, *theta)

        bnds = ((-np.inf, np.inf), (0, np.inf), (0, np.inf))
        init_theta = [np.mean(x), 0.05, 1]

        sol = minimize(objfun, init_theta, method='Nelder-Mead', bounds=bnds)
        self.sol = sol

        return sol


class GeneralizedT(Distribution):

    def __init__(self) -> None:

        super().__init__()
        self.num_parameters = 4
        self.parameters = ['mu', 'sigma', 'nu', 'tau']
        self.name = 'Generalized T'

    @staticmethod
    def pdf(x, mu, sigma, nu, tao):
        return (tao / (2*sigma*nu**(1/tao) * beta(nu, 1/tao)))*(1 + 1/nu * np.abs((x-mu)/sigma)**tao)**(-(nu+1/tao))

    def log_likelihood(self, x, mu, sigma, nu, tao):
        return np.sum(np.log(self.pdf(x, mu, sigma, nu, tao)))

    def mle(self, x):
        super().mle(x)
        def objfun(theta): return -1*self.log_likelihood(x, *theta)

        bnds = ((-np.inf, np.inf), (0.001, np.inf), (0.001, np.inf),
                (0.001, np.inf))  # bounds on parameters

        init_theta = [np.mean(x), np.std(x, ddof=1), 1, 1]

        sol = minimize(objfun, init_theta, method='Nelder-Mead', bounds=bnds)
        self.sol = sol

        return sol


class NormalizedInverseGaussian(Distribution):
    """
    There is a normal inverse gaussian pdf function in scipy, which references the same source as the paper
    with  a = alpha * delta, b = beta * delta, loc = mu, scale=delta according to scipy documentation

    There is a requirement in scipy that |beta|<=alpha, this corresponds to |beta|<alpha, which isn't stated explicitly in the paper
    but is a requirement in order to ensure their paramter gamma=sqrt(alpha**2-beta**2) is real
    """

    def __init__(self) -> None:

        super().__init__()
        self.num_parameters = 4
        self.parameters = ['mu', 'delta', 'alpha', 'beta']
        self.name = 'Normalized Inverse Gaussian'

    @staticmethod
    def pdf(x, mu, delta, alpha, beta):
        return stats.norminvgauss.pdf(x, a=alpha*delta, b=beta*delta, loc=mu, scale=delta)

    def log_likelihood(self, x, mu, delta, alpha, beta):
        return np.sum(np.log(self.pdf(x, mu, delta, alpha, beta)))

    def mle(self, x):
        super().mle(x)
        def objfun(theta): return -1*self.log_likelihood(x, *theta)

        bnds = ((-np.inf, np.inf), (0.01, np.inf), (0.01, np.inf),
                (0.01, np.inf))  # bounds on parameters

        # inequality constraint alpha-beta>0 equivalent to alpha>beta
        def constraint(theta): return theta[2]-theta[3]

        con = {'type': 'ineq', 'fun': constraint}

        init_theta = [np.mean(x), np.std(x, ddof=1), 1, 0.5]

        sol = minimize(objfun, init_theta, method='Nelder-Mead',
                       bounds=bnds, constraints=con)
        self.sol = sol

        return sol


class GeneralizedHyperbolic(Distribution):

    def __init__(self) -> None:

        super().__init__()
        self.num_parameters = 5
        self.parameters = ['mu', 'delta', 'lambda', 'alpha', 'beta']
        self.name = 'Generalized Hyperbolic'

    @staticmethod
    def pdf(x, mu, delta, lmda, alpha, beta):
        return stats.genhyperbolic.pdf(x, p=lmda, a=delta*alpha, b=beta*delta, scale=delta, loc=mu)

    def log_likelihood(self, x, mu, delta, lmda, alpha, beta):
        return np.sum(np.log(self.pdf(x, mu, delta, lmda, alpha, beta)))

    def mle(self, x):
        super().mle(x)
        def objfun(theta): return -1*self.log_likelihood(x, *theta)

        bnds = ((-np.inf, np.inf), (0.01, np.inf), (-np.inf, np.inf),
                (0.01, np.inf), (0.01, np.inf))  # bounds on parameters

        # inequality constraint alpha-beta>0 equivalent to alpha>beta
        def constraint(theta): return theta[3]-theta[4]

        con = {'type': 'ineq', 'fun': constraint}

        init_theta = [np.mean(x), np.std(x, ddof=1), 1, 1, 0.5]

        sol = minimize(objfun, init_theta, method='Nelder-Mead',
                       bounds=bnds, constraints=con)
        self.sol = sol

        return sol


class SkewT(Distribution):

    def __init__(self) -> None:

        super().__init__()
        self.num_parameters = 4
        self.parameters = ['mu', 'sigma', 'nu', 'lambda']
        self.name = 'Skew T'

    @staticmethod
    def pdf(x, mu, sigma, nu, lmda):
        output = (2/sigma)*stats.t.pdf((x-mu)/sigma, df=nu) *\
            stats.t.cdf((lmda * (x-mu)/sigma) *
                        np.sqrt((nu+1) / (((x-mu)/sigma)**2 + nu)), df=nu+1)

        return output

    def log_likelihood(self, x, mu, sigma, nu, lmda):
        return np.sum(np.log(self.pdf(x, mu, sigma, nu, lmda)))

    def mle(self, x):
        super().mle(x)
        def objfun(theta): return -1*self.log_likelihood(x, *theta)

        bnds = ((-np.inf, np.inf), (1e-15, np.inf), (1e-15, np.inf),
                (-np.inf, np.inf))  # bounds on parameters

        init_theta = [np.mean(x), np.std(x, ddof=1), 1, 1]

        sol = minimize(objfun, init_theta, method='Nelder-Mead', bounds=bnds)
        self.sol = sol

        return sol


class SkewedStudent(Distribution):

    def __init__(self) -> None:

        super().__init__()
        self.num_parameters = 4
        self.parameters = ['mu', 'sigma', 'nu', 'alpha']
        self.name = 'Skewed Student T'

    def pdf(self, x, mu, sigma, nu, alpha):
        indicator = (x > mu)*1  # switch on if x is greater than mu
        output = (self.kappa(nu)/sigma) *\
            (1+(1/nu)*((x-mu)/(2*sigma*(alpha*(1-indicator) + (1-alpha)*indicator)))**2)**(-(nu+1)/2)
        return output

    def log_likelihood(self, x, mu, sigma, nu, alpha):
        return np.sum(np.log(self.pdf(x, mu, sigma, nu, alpha)))

    def mle(self, x):
        super().mle(x)
        def objfun(theta): return -1*self.log_likelihood(x, *theta)

        bnds = ((-np.inf, np.inf), (1e-15, np.inf), (1e-15, np.inf),
                (-np.inf, np.inf))  # bounds on parameters

        init_theta = [np.mean(x), np.std(x, ddof=1), 1, 0.5]

        sol = minimize(objfun, init_theta, method='Nelder-Mead', bounds=bnds)
        self.sol = sol

        return sol


class AsymmetricStudentT(Distribution):

    def __init__(self) -> None:

        super().__init__()
        self.num_parameters = 5
        self.parameters = ['mu', 'sigma', 'nu1', 'nu2', 'alpha']
        self.name = 'Asymmetric Student T'

    def pdf(self, x, mu, sigma, nu1, nu2, alpha):
        indicator = (x > mu)*1  # switch on if x is greater than mu

        alpha2 = alpha*self.kappa(nu1)/(alpha *
                                        self.kappa(nu1)+(1-alpha)*self.kappa(nu2))

        nu_indicator = nu1*(1-indicator) + nu2*indicator

        output = ((alpha/alpha2*(1-indicator) + (1-alpha)/(1-alpha2)*indicator) * self.kappa(nu_indicator)/sigma *
                  (1+1/nu_indicator*((x-mu)/(2*sigma*(alpha2*(1-indicator) + (1-alpha2)*indicator)))**2)**(-(nu_indicator+1)/2))
        return output

    def log_likelihood(self, x, mu, sigma, nu1, nu2, alpha):
        return np.sum(np.log(self.pdf(x, mu, sigma, nu1, nu2, alpha)))

    def mle(self, x):
        super().mle(x)
        def objfun(theta): return -1*self.log_likelihood(x, *theta)

        bnds = ((-np.inf, np.inf), (0.001, np.inf), (0.001, np.inf),
                (0.001, np.inf), (0.01, 0.99))  # bounds on parameters

        init_theta = [np.mean(x), np.std(x, ddof=1), 1, 1, 0.5]

        sol = minimize(objfun, init_theta, method='Nelder-Mead', bounds=bnds)

        self.sol = sol
        return sol
