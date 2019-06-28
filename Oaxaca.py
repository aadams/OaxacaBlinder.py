import pandas as pd 
import numpy as np 
import statsmodels.api as sm
import matplotlib.pyplot as plt
class Oaxaca:

    def __init__(self, data, by, endo):
        
        self.data = data
        self.by = by
        self.df_type = ""
        self.f_df = ""
        self.s_df = ""
        self.endo = endo
        self.two_gap = 0
        self.three_gap = 0

        self.f_mean = 0
        self.s_mean = 0

        self.char_eff = 0
        self.coef_eff = 0
        self.int_eff = 0

        self.exp_eff = 0
        self.unexp_ex = 0

        self.t_x = 0
        self.t_y = 0

        self.explained = 0
        self.unexplained = 0

        #A bunch of error checking
        if type(self.data) != type(pd.DataFrame()) and type(self.data) != type(np.array(1)):
            raise ValueError('The data must be in a DataFrame or a numpy array')


        if type(self.data) == type(pd.DataFrame()):
            #By must be a string
            if type(self.by) != str:
                raise ValueError('The "by" variable must be a string if datatype is {}'.format(type(self.data)))

            if type(self.endo) != str:
                raise ValueError('The "endo" variable must be a string if datatype is {}'.format(type(self.data)))
            
            #The by is not in the columns
            if by not in self.data.columns.values:
                raise ValueError('The "by" variable must be in the DataFrame')


            if endo not in self.data.columns.values:
                raise ValueError('The "endo" variable must be in the DataFrame')

            self.df_type = 'df'

        if type(self.data) == type(np.array(1)):
            #By must be an integer to index the numpy array
            if type(self.by) != int:
                raise ValueError('The "by" variable must be a int if datatype is {}'.format(type(self.data)))

            if type(self.by) != int:
                raise ValueError('The "endo" variable must be a int')

            self.df_type = 'np'

        #Split the Dataframe by the 'By' value
        if self.df_type == 'np':
            self.data = pd.DataFrame(self.data)
            split = self.data.iloc[:,by].value_counts().index
            
            #We need binary differences for this value
            if len(split) != 2:
                print("These are the attempted split values: {}".format(split))
                raise KeyError('There are more than 2 unique values in the by columns')

            print("These are the attempted split values: {}".format(split))
            
            self.t_x = self.data.drop(self.data.iloc[:, endo], axis = 1)
            self.t_y = self.data.iloc[:, endo]

            self.f_df = self.data[self.data.iloc[:, by] == split[0]]
            self.s_df = self.data[self.data.iloc[:, by] != split[0]]
           
            self.f_x = self.f_df.drop(self.f_df.iloc[:, [by, endo]], axis = 1)
            self.f_y = self.f_df.iloc[:, endo]
        
            self.s_x = self.s_df.drop(self.s_df.iloc[:, [by, endo]], axis = 1)
            self.s_y = self.s_df.iloc[:, endo]

            self.f_x = sm.add_constant(self.f_x)
            self.s_x = sm.add_constant(self.s_x)
            self.t_x = sm.add_constant(self.t_x)


        if self.df_type == 'df':
            split = self.data[by].value_counts().index
            
            if len(split) != 2:
                print("These are the attempted split values: {}".format(split))
                raise KeyError('There are more than 2 unique values in the by columns')

            print("These are the attempted split values: {}".format(split))
            
            self.t_x = self.data.drop([endo], axis = 1)
            self.t_y = self.data[endo]


            self.f_df = self.data[self.data[by] == split[0]]
            self.s_df = self.data[self.data[by] != split[0]]

            self.f_x = self.f_df.drop([by,endo], axis = 1)
            self.f_y = self.f_df[endo]

            self.s_x = self.s_df.drop([by,endo], axis = 1)
            self.s_y = self.s_df[endo]

            self.f_x = sm.add_constant(self.f_x)
            self.s_x = sm.add_constant(self.s_x)
            self.t_x = sm.add_constant(self.t_x)


    def fix(self):
        #There may be issues with the "first" dataframe not being the correct one
        #we remidy this by flipping the two if the gap is negative
        self.f_df, self.s_df = self.s_df, self.f_df
        self.f_x, self.s_x = self.s_x, self.f_x
        self.f_y, self.s_y = self.s_y, self.f_y
        self.f_mean, self.s_mean = self.s_mean, self.f_mean
        
    def three_fold(self, plot = False):
        self.f_mean = self.f_y.mean()
        self.s_mean = self.s_y.mean()
        
        #The wrong first is first
        if self.f_mean - self.s_mean < 0:
            self.fix()
        
        self.f_model = sm.OLS(self.f_y, self.f_x).fit()
        self.s_model = sm.OLS(self.s_y, self.s_x).fit()

        #Characteristic Effect
        self.char_eff = (self.f_x.mean() - self.s_x.mean()) @ self.s_model.params

        #Coefficient Effect
        self.coef_eff = (self.s_x.mean()) @ (self.f_model.params - self.s_model.params)

        #Interaction Effect
        self.int_eff = (self.f_x.mean() - self.s_x.mean()) @ (self.f_model.params - self.s_model.params)
        
        self.three_gap = self.f_mean - self.s_mean

        print("Characteristic Effect: {}".format(self.char_eff))
        print("Coefficent Effect: {}".format(self.coef_eff))
        print("Interaction Effect: {}".format(self.int_eff))
        print("Gap: {}".format(self.three_gap))
        
        if plot == True:
            self.plot(plt_type=3)

        return self.char_eff, self.coef_eff, self.int_eff, self.three_gap

    def two_fold(self, plot = False):
        self.f_mean = self.f_y.mean()
        self.s_mean = self.s_y.mean()
        
        #The wrong first is first
        if self.f_mean - self.s_mean < 0:
            self.fix()

        self.t_model = sm.OLS(self.t_y, self.t_x).fit()
        self.t_params = self.t_model.params.drop(self.by)
        self.f_model = sm.OLS(self.f_y, self.f_x).fit()
        self.s_model = sm.OLS(self.s_y, self.s_x).fit()
            
        self.unexplained = (self.f_x.mean() @ (self.f_model.params - self.t_params)) + (self.s_x.mean() @ (self.t_params - self.s_model.params))

        self.explained = (self.f_x.mean() - self.s_x.mean()) @ self.t_params
        
        self.two_gap = self.f_mean - self.s_mean

        print('Unexplained Effect: {}'.format(self.unexplained))
        print('Explained Effect: {}'.format(self.explained))
        print('Gap: {}'.format(self.two_gap))
        if plot == True:
            self.plot(plt_type = 2)

        return self.unexplained, self.explained, self.two_gap

    def var(self):
        #TODO
        #Need to add some sort of t-test
        return ''
    
    def cotton_fix(self):
        #This adjusts for over representation
        #TODO
        return ''

    def plot(self, plt_type = 3, fig_size = (6,10), xlabel = '', ylabel = 'Oaxaca Values', color1 = 'seagreen', color2 = 'darkturquoise', color3 = 'steelblue', color4 = 'navy'):
        #the plot types must either be able to made into an int or be an int
        try:
            plt_type = int(plt_type)
        except ValueError:
            raise ValueError('The plot type must be an integer.')

        #we only have two types of plots, 3 or 2, so it must be one of the two
        if plt_type != 3 and plt_type != 2:
            raise ValueError("The plot types must be two or three")

        #all the colors and labels must be strings
        if any(map((lambda value: type(value) != str), (xlabel, ylabel, color1, color2, color3, color4))):
            raise ValueError('All labels and colors must be strings.')
        
        #This sets the xlabel if default
        if xlabel == '':
            if plt_type == 3:
                xlabel = 'Three-Fold Oaxaca Plot'
            elif plt_type == 2:
                xlabel = 'Two-Fold Oaxaca Plot'

        if plt_type == 2:
            if self.explained == 0 and self.unexplained == 0:
                raise ValueError("Please fit the values before attempting to plot")
                
        if plt_type == 3:
            if self.char_eff == 0 and self.coef_eff == 0:
                raise ValueError("Please fit the values before attempting to plot")
        
        #this is the three_fold plot
        if plt_type == 3:
            fig, ax = plt.subplots(figsize = fig_size)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.bar(x= 0, height = self.char_eff, width = .25, label = 'Character Effect', color = color1)
            plt.bar(x=0, height = self.coef_eff, bottom= self.char_eff, width = .25, label = 'Coefficent Effect', color = color2)
            plt.bar(x=0, height = -self.int_eff, width = .25, label = 'Interaction Effect', color = color3)
            plt.bar(x = .25, height = self.three_gap, width = .25, label = 'Total Gap', color = color4)
            plt.ylim(top = self.three_gap + .15)
            plt.xlim([-.2,.5])
            plt.axhline(y=0, color = 'k', linestyle = '--')
            ax.grid(zorder=0)
            plt.legend()

        #this is a two_fold plot
        if plt_type == 2:
            fig, ax = plt.subplots(figsize = fig_size)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.bar(x= 0, height = self.unexplained, width = .25, label = 'Explained', color = color1)
            plt.bar(x=0, height = self.explained, bottom= self.unexplained, width = .25, label = 'Unexplained', color = color2)
            plt.bar(x = .25, height = self.two_gap, width = .25, label = 'Total Gap', color = color3)
            plt.ylim(top = self.two_gap + .15)
            plt.xlim([-.2,.5])
            plt.axhline(y=0, color = 'k', linestyle = '--')
            ax.grid(zorder=0)
            plt.legend()