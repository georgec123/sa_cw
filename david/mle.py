import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import numpy as np
from scipy.optimize import minimize
import scipy.stats as stats
from scipy.special import hyp2f1, beta #gauss hypergeometric functi
from abc import ABC, abstractproperty, abstractmethod
from scipy.special import beta, hyp2f1


class Distribution(ABC):
    def __init__(self) -> None:

        self.sol = None
        self.name = None
        self.num_parameters  = None
        self.parameters = None
        return

    
    def print_params(self):
        for idx, param in enumerate(self.parameters):
            print(f"{param}: {self.sol.x[idx]}")

    @abstractmethod
    def log_likelihood(x):
        pass

    @abstractmethod
    def mle(x):
        pass

        
    def aic(self):
        '''
        The AIC of an MLE estiamte equals 2*number of parameters - value of loglikelihood
        Inputs:
        k-number of parameters in distribution
        L-the value of the loglikelihood function
        '''
        return 2*self.num_parameters +2*self.sol.fun


    def kappa(self, nu):
        return 1/(np.sqrt(nu)*beta(nu/2, 1/2))

class Laplace(Distribution):

    def __init__(self) -> None:

        super().__init__()
        self.num_parameters = 2
        self.parameters = ['mu', 'b']
        self.name = 'Laplace'
        

    @staticmethod
    def log_likelihood(mu,b,x):
        n=len(x)
        return -n*np.log(2*b)-np.sum(np.abs(x-mu)/b)

    def mle(self, x):
        objfun=lambda theta : -1*self.log_likelihood(theta[0],theta[1],x) 
        bnds=((-np.inf,np.inf), (1e-15,np.inf)) #bounds on parameters

        init_theta=[np.mean(x),1]

        sol=minimize(objfun,init_theta, method='Nelder-Mead',bounds=bnds)
        
        self.sol = sol
        return sol.x
    
class StudentT(Distribution):
    def __init__(self) -> None:

        super().__init__()
        self.num_parameters = 3
        self.parameters = ['mu', 'sigma', 'nu']
        self.name = 'Student T'

       
    def student_t_pdf(self, x,mu,sigma,nu):

        return (self.kappa(nu)/sigma) * (1+ ((x-mu)**2)/(nu * sigma**2))**(-(1+nu)/2)


    
    def log_likelihood(self, mu,sigma,nu, x):
        return np.sum(np.log(self.student_t_pdf(x,mu,sigma,nu)))

    def mle(self, x):
        objfun= lambda theta: -1*self.log_likelihood(theta[0],theta[1],theta[2], x)

        bnds=((-np.inf,np.inf), (0,np.inf), (0,np.inf) ) 
        init_theta=[np.mean(x),0.05,1]

        sol=minimize(objfun,init_theta, method='Nelder-Mead',bounds=bnds)

        self.sol = sol

        return sol

class GeneralizedT(Distribution):

    def __init__(self) -> None:

        super().__init__()
        self.num_parameters = 4
        self.parameters = ['mu', 'sigma', 'nu', 'tau']
        self.name = 'Generalized T'
        
    @staticmethod
    def generalized_t_pdf(x,mu,sigma,nu,tao):
        return (tao/ (2*sigma*nu**(1/tao) * beta(nu,1/tao)))*(1+ 1/nu * np.abs((x-mu)/sigma)**tao)**(-(nu+1/tao))

    def log_likelihood(self, mu,sigma,nu,tao,x):
        return np.sum(np.log(self.generalized_t_pdf(x,mu,sigma,nu,tao)))


    def mle(self, x):
        objfun= lambda theta: -1*self.log_likelihood(theta[0],theta[1],theta[2],theta[3],x)
        
        bnds=((-np.inf,np.inf), (0.001,np.inf), (0.001,np.inf), (0.001,np.inf) ) #bounds on parameters

        init_theta=[np.mean(x),np.std(x,ddof=1),1,1]

        sol=minimize(objfun,init_theta, method='SLSQP',bounds=bnds)

        self.sol = sol
        return sol

    '''
There is a normal inverse gaussian pdf function in scipy, which references the same source as the paper
with  a = alpha * delta, b = beta * delta, loc = mu, scale=delta according to scipy documentation

There is a requirement in scipy that |beta|<=alpha, this corresponds to |beta|<alpha, which isn't stated explicitly in the paper
but is a requirement in order to ensure their paramter gamma=sqrt(alpha**2-beta**2) is real
'''

class NormalizedInverseGaussian(Distribution):

    def __init__(self) -> None:

        super().__init__()
        self.num_parameters = 4
        self.parameters = ['mu', 'delta', 'alpha', 'beta']
        self.name = 'Normalized Inverse Gaussian'

    @staticmethod
    def log_likelihood(mu,delta,alpha,beta,x):
        return np.sum(np.log(stats.norminvgauss.pdf(x,a=alpha*delta,b=beta*delta,loc=mu,scale=delta)))

    def mle(self, x):
        objfun= lambda theta: -1*self.log_likelihood(theta[0],theta[1],theta[2],theta[3],x)
        
        bnds=((-np.inf,np.inf), (0.01,np.inf), (0.01,np.inf), (0.01,np.inf) ) #bounds on parameters
        
        constraint= lambda theta : theta[2]-theta[3] #inequality constraint alpha-beta>0 equivalent to alpha>beta
        
        con={'type': 'ineq','fun':constraint}

        init_theta=[np.mean(x),np.std(x,ddof=1),1,0.5]

        sol=minimize(objfun,init_theta, method='SLSQP',bounds=bnds,constraints=con)

        self.sol = sol
        return  sol



class GeneralizedHyperbolic(Distribution):

    def __init__(self) -> None:

        super().__init__()
        self.num_parameters = 5
        self.parameters = ['mu', 'delta', 'lambda', 'alpha' ,'beta']
        self.name = 'Generalized Hyperbolic'

    @staticmethod
    def log_likelihood(mu,delta,lmda,alpha,beta,x):
        return np.sum(np.log(stats.genhyperbolic.pdf(x,p=lmda,a=delta*alpha,b=beta*delta,scale=delta,loc=mu)))

    def mle(self, x):
        objfun= lambda theta: -1*self.log_likelihood(theta[0],theta[1],theta[2],theta[3],theta[4],x)
        
        bnds=((-np.inf,np.inf), (0.01,np.inf), (-np.inf,np.inf), (0.01,np.inf), (0.01,np.inf) ) #bounds on parameters
        
        constraint= lambda theta : theta[3]-theta[4] #inequality constraint alpha-beta>0 equivalent to alpha>beta
        
        con={'type': 'ineq','fun':constraint}

        init_theta=[np.mean(x),np.std(x,ddof=1),1,1,0.5]

        sol=minimize(objfun,init_theta, method='Nelder-Mead',bounds=bnds,constraints=con)

        self.sol = sol

        return sol




class SkewT(Distribution):

    def __init__(self) -> None:

        super().__init__()
        self.num_parameters = 4
        self.parameters = ['mu', 'sigma', 'nu', 'lambda']
        self.name = 'Skew T'
        
    @staticmethod
    def skewt_pdf(x,mu,sigma,nu,lmda):
        return (2/sigma)*stats.t.pdf((x-mu)/sigma,df=nu)*\
            stats.t.cdf((lmda*(x-mu)/sigma) *np.sqrt( (nu+1)/(((x-mu)/sigma)**2 + nu) ) ,df=nu+1)

    # def skewt_pdf(self, x,mu,sigma,nu,lmda):
    #     a = sigma*stats.t.pdf((x-mu)/sigma,df=nu)
    #     b = 2*lmda*(x-mu)*(self.kappa(nu)**2)/(sigma**2)
    #     c = hyp2f1(0.5, (1+nu)/2, 3/2, -((lmda*(x-mu)/sigma)**2)/nu)
        
    #     return a+b*c
    
    # @staticmethod
    # def skewt_pdf(self, x,mu,sigma,nu,lmda):
    #     # b = 2*lmda*(x-mu)*(self.k(nu)**2)/(sigma**2)
    #     # c = hyp2f1(0.5, (1+nu)/2, 3/2, -((lmda*(x-mu)/sigma)**2)/nu)
    #     # return sigma*stats.t.pdf(x-mu/sigma,df=nu) + b*c
    #     #     # stats.t.cdf(lmda* (x-mu)/sigma *np.sqrt( (nu+1)/(((x-mu)/sigma)**2 + nu) ) ,df=nu+1)
    #     dist = SST(mu = mu, sigma = sigma, nu = nu, tau = 5)


    def log_likelihood(self, mu,sigma,nu, lmda,x):
        return np.sum(np.log(self.skewt_pdf(x,mu,sigma,nu,lmda)))

    def mle(self, x):
        objfun= lambda theta: -1*self.log_likelihood(theta[0],theta[1],theta[2],theta[3],x)

        bnds=((-np.inf,np.inf), (1e-15,np.inf), (1e-15,np.inf), (-np.inf,np.inf) ) #bounds on parameters

        init_theta=[np.mean(x),np.std(x,ddof=1),1,1]

        sol=minimize(objfun,init_theta, method='Nelder-Mead',bounds=bnds)

        self.sol = sol

        return sol



class SkewedStudent(Distribution):

    def __init__(self) -> None:

        super().__init__()
        self.num_parameters = 4
        self.parameters = ['mu', 'sigma', 'nu', 'alpha']
        self.name = 'Skewed Student T'
    
    
    def skewed_student_t_pdf(self, x,mu,sigma,nu,alpha):
        indicator=(x>mu)*1 #switch on if x is greater than mu
        return (self.kappa(nu)/sigma) * (1+(1/nu)*( (x-mu)/(2*sigma*( alpha*(1-indicator) +(1-alpha)*indicator) ) )**2)**(-(nu+1)/2)

    def log_likelihood(self, mu,sigma,nu,alpha,x):
        return np.sum(np.log(self.skewed_student_t_pdf(x,mu,sigma,nu,alpha)))

    def mle(self, x):
        objfun= lambda theta: -1*self.log_likelihood(theta[0],theta[1],theta[2],theta[3],x)
        
        bnds=((-np.inf,np.inf), (0.01,np.inf), (0.01,np.inf), (0.0001,0.9999) ) #bounds on parameters

        init_theta=[np.mean(x),np.std(x,ddof=1),1,0.5]

        sol = minimize(objfun,init_theta, method='SLSQP',bounds=bnds)

        self.sol = sol
        return sol

    
class AsymmetricStudent(Distribution):
    def __init__(self) -> None:

        super().__init__()
        self.num_parameters = 5
        self.parameters = ['mu', 'sigma', 'nu1','nu2', 'alpha']
        self.name = 'Asymmetric Student T'
        
    def asym_student_t_pdf(x,mu,sigma,nu1,nu2,alpha):
    indicator=(x>mu)*1 #switch on if x is greater than mu
    
    alpha2=alpha*Kappa(nu1)/(alpha*Kappa(nu1)+(1-alpha)*Kappa(nu2))
    
    nu_indicator=nu1*(1-indicator) + nu2*indicator
    
    return ((alpha/alpha2*(1-indicator) + (1-alpha)/(1-alpha2)*indicator) *
            Kappa(nu_indicator)/sigma*\
            (1+1/nu_indicator*( (x-mu)/(2*sigma*( alpha2*(1-indicator) +(1-alpha2)*indicator) ) )**2)**(-(nu_indicator+1)/2))

    def log_likelihood(self,mu,sigma,nu1,nu2,alpha,x):
    return np.sum(np.log(asym_student_t_pdf(x,mu,sigma,nu1,nu2,alpha)))

    def mle(self,x):
    objfun= lambda theta: -1*asym_student_loglikelihood(theta[0],theta[1],theta[2],theta[3],theta[4],x)
    
    bnds=((-np.inf,np.inf), (0.001,np.inf), (0.001,np.inf), (0.001,np.inf), (0.01,0.99) ) #bounds on parameters

    init_theta=[np.mean(x),np.std(x,ddof=1),1,1,0.5]

    sol=minimize(objfun,init_theta, method='SLSQP',bounds=bnds)
    return sol