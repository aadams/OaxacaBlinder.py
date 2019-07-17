#TODO Variance can be calculated for the three_fold
#TODO Group Size Effects can be accounted for
#TODO Non-Linear Oaxaca-Blinder can be used
"""
*************************************
Created on July 16  2018

Author: Austin Adams
**************************************
This Class implements Oaxaca-Blinder Decomposition:

Two-Fold (two_fold)
Three-Fold (three_fold)

A Oaxaca-Blinder is a statistical method that is used to explain
the differences between two mean values. The idea is to show
from two mean values what can be explained by the data and 
what cannot by using OLS regression frameworks.

"The original use by Oaxaca's was to explain the wage 
differential between two different groups of workers, 
but the method has since been applied to numerous other topics." (Wikipedia)


The model is designed to accept two endogenous response variables
and two exogenous explanitory variables. They are then fit using
the specific type of decomposition that you want.


General reference for Oaxaca-Blinder:

B. Jann "The Blinder-Oaxaca decomposition for linear
regression models," The Stata Journal, 2008.

Econometrics references for regression models:

E. M. Kitagawa  "Components of a Difference Between Two Rates" 
Journal of the American Statistical Association, 1955.

A. S. Blinder "Wage Discrimination: Reduced Form and Structural
Estimates," The Journal of Human Resources, 1973.
"""
import statsmodels.api as sm
import numpy as np
class Oaxaca:
__doc__ = r"""
    Prepares to perform Oaxaca-Blinder Decomposition.

    %(params)s
    endo: array-like
        'endo' is the endogenous variable or the dependent variable 
        that you are trying to explain.  
    exo: array-like
        'exo' is the exogenous variable(s) or the independent variable(s) 
        that you are using to explain the endogenous variable.
    bifurcate: int
        'bifurcate' is the column of the exogenous variable(s) that you 
        wish to split on. This would generally be the group that you wish
        to explain the two means for.
    hasconst: None or bool
        Indicates whether the two exogenous variables include a user-supplied
        constant. If True, a constant is assumed. If False, a constant is added
        at the start. If nothing is supplied, then True is assumed.

    Attributes
    ----------
    None

    Methods:
    ----------
    three_fold()
        Returns the three-fold decomposition of Oaxaca-Blinder

    two_fold()
        Returns the two-fold decomposition of the Oaxaca-Blinder
        
    Notes
    -----
    Please check if your data includes at constant. This will still run, but
    will return extremely incorrect values if set incorrectly.

    Examples
    --------
    >>> import numpy as np
    >>> import statsmodels.api as sm
    >>> import Oaxaca
    >>> data = sm.datasets.ccards.load()
    
    '3' is the column of which we want to explain or which indicates
    the two groups. In this case, it is if you rent.
    
    >>> model = Oaxaca(df.endog, df.exog, 3, hasconst = False)
    >>> model.two_fold()
    >>> ******************************
        Unexplained Effect: 27.94091
        Explained Effect: 130.80954
        Gap: 158.75044
        ******************************
    >>> model.three_fold()
    >>> ******************************
        Characteristic Effect: 321.74824
        Coefficent Effect: 75.45371
        Interaction Effect: -238.45151
        Gap: 158.75044
        ******************************
    """
import statsmodels.api as sm
import numpy as np
class Oaxaca:
    def __init__(self, endo, exo, bifurcate, hasconst = True):
            #This does most of the big calculations
            bi_col = exo[:, bifurcate]
            endo = np.append(bi_col.reshape(-1,1), endo.reshape(-1,1), axis = 1)
            bi = np.unique(bi_col)
            
            exo_f = exo[np.where(exo[:, bifurcate] == bi[0])]
            exo_s = exo[np.where(exo[:, bifurcate] == bi[1])]
            endo_f = endo[np.where(endo[:, 0] == bi[0])]
            endo_s = endo[np.where(endo[:, 0] == bi[1])]
            exo_f = np.delete(exo_f, bifurcate, axis = 1)
            exo_s = np.delete(exo_s, bifurcate, axis = 1)
            endo_f = endo_f[:,1]
            endo_s = endo_s[:,1]
            endo = endo[:,1]
            
            self.gap = endo_f.mean() - endo_s.mean() 
            
            if self.gap < 0:
                endo_f, endo_s = endo_s, endo_f
                exo_f, exo_s = exo_s, exo_f
                self.gap = endo_f.mean() - endo_s.mean()
            
            if hasconst == False:
                exo_f = sm.add_constant(exo_f, prepend = False)
                exo_s = sm.add_constant(exo_s, prepend = False)
                exo = sm.add_constant(exo, prepend = False)
            
            self._t_model = sm.OLS(endo, exo).fit()
            self._f_model = sm.OLS(endo_f, exo_f).fit()
            self._s_model = sm.OLS(endo_s, exo_s).fit()
            
            self.exo_f_mean = np.mean(exo_f, axis = 0)
            self.exo_s_mean = np.mean(exo_s, axis = 0)
            self.t_params = np.delete(self._t_model.params, bifurcate)
        
    def three_fold(self):
        """
        Calculates the three-fold Oaxaca Blinder Decompositions

        Parameters
        ----------

        None

        Returns
        -------

        char_eff : float
            This is the effect due to the group differences in
            predictors

        coef_eff: float
            This is the effect due to differences of the coefficients
            of the two groups
        
        int_eff: float
            This is the effect due to differences in both effects
            existing at the same time between the two groups.
        """
        #Characteristic Effect
        self.char_eff = (self.exo_f_mean - self.exo_s_mean) @ self._s_model.params
        #Coefficient Effect
        self.coef_eff = (self.exo_s_mean) @ (self._f_model.params - self._s_model.params)
        #Interaction Effect
        self.int_eff = (self.exo_f_mean - self.exo_s_mean) @ (self._f_model.params - self._s_model.params)
        
        print("".join(["*" for x in range(0,30)]))
        print("Characteristic Effect: {:.5f}\nCoefficent Effect: {:.5f}\nInteraction Effect: {:.5f}\nGap: {:.5f}".format(self.char_eff, self.coef_eff, self.int_eff, self.gap))
        print("".join(["*" for x in range(0,30)]))
        
        return self.char_eff, self.coef_eff, self.int_eff, self.gap
    
    def two_fold(self):
        """
        Calculates the three-fold Oaxaca Blinder Decompositions

        Parameters
        ----------

        None

        Returns
        -------

        unexplained : float
            This is the effect that cannot be explained by the data at hand.
            This does not mean it cannot be explained with more.
        
        explained: float
            This is the effect that can be explained using the data.
        """
        #Unexplained Effect
        self.unexplained = (self.exo_f_mean @ (self._f_model.params - self.t_params)) + (self.exo_s_mean @ (self.t_params - self._s_model.params))
        #Explained Effect
        self.explained = (self.exo_f_mean - self.exo_s_mean) @ self.t_params
        
        print("".join(["*" for x in range(0,30)]))
        print("Unexplained Effect: {:.5f}\nExplained Effect: {:.5f}\nGap: {:.5f}".format(self.unexplained, self.explained, self.gap))        
        print("".join(["*" for x in range(0,30)]))
        
        return self.unexplained, self.explained, self.gap