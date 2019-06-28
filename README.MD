# Oaxaca.py

This package attempts to port the functionality of the oaxaca command in STATA to Python. The Oaxaca-Blinder, or Blinder-Oaxaca as some call it,
decomposition attempts to explain gaps in means of groups. It uses the linear models of two given regression equations to show what is explained by regression coefficients and known data and
what is unexplained using the data. There are two types of Oaxaca-Blinder decompositions, the two-fold and the three-fold,
both of which can and are used in Economics Literature to discuss differences in groups. If you would like to learn more about the specifics, read the journal article by Dr. Ben Jann in the papers folder of the repo.
 
### Prerequisites

This package requires NumPy, Pandas, and StatsModels. If you intend to use the plotting feature, you also need Matplotlib.


### Installing

To install the package, simply download the Oaxaca.py file, place it in the working directory, and call from Oaxaca import Oaxaca.

```
from Oaxaca import Oaxaca
```

## Usage

This package has four current commands with the intend to add more later.

The current are fit, plot, var, and cotton_model.

### The Oaxaca object

The Oaxaca object is called when you use the code below.

```
model = Oaxaca(data, by, endo, debug)
```
data is required to be either a Numpy array or a Pandas DataFrame. The by value shows which column to bifurcate the date, and the endo value shows which column you wish to explain. debug is set to True by default if you would like error-checking of your data to occur.

These are the needed data types depending on what type your data is in. 

data type| Numpy Array | Pandas DataFrame |
---| --- | --- |
by type| integer | string |
endo type| integer | string |


### fit

This calculates the three_fold decomposition and the two_fold decomposition as described in the Jann paper.

```
characteristic, coefficient, interaction, gap = model.fit(two_fold = False, three_fold = True, plot, round_val)
unexplained, explained, gap = model.fit(two_fold = True, three_fold = False, plot, round_val)
```
This function returns values, all floats, of the known effects and then the differences in means.

If plot, which can be set to True or False, is set to True, a plot of the graphs will be returned with the default settings. round_val takes either integers or variables that can be casted to integers and rounds the values of the effects to avoid floating point errors. This can be turned off by saying round_val = False.

If both are selected as True, the two_fold will be calculated first, then three_fold. Fitting is required to do any of the other commands.


## plot

This function draws a customizable plot of the effects on the screen

```
model.plot(plt_type, fig_size, xlabel, ylabel, color1, color2, color3, color4)
```

The plot, while allowing for custom settings, does not need them. The following chart will depict what is needed for these values.


Setting| Description | Variable Type | Required
---| --- | --- |
plt_type| chooses between two_fold, three_fold, and cotton_model | integer | defaults to 3
fig_size | chooses size of plot | tuple of integers | defaults to [6,10]
xlabel/ylabel| chooses the labels | string | defaults to the type of plot
color(1,2,3,4) | chooses colors of the plot | string | defaults to a cool pallet

If you wish to just use default settings, feel free to leave the values blank. 

## var

This function calculates the variance of the three_fold decomposition

```
characteristic variance, coefficient variance = model.var(round_val)
```
This takes a rounding value, similar to fit, which can be stopped by round_val = False

## cotton_model

This function uses the adjusted in Cotton (1988), which accounts for the undervaluation of one group causing the overevalution of another.

```
unexplained, explained, gap = model.cotton_model(plot, round_val)
```

This takes plot and round_val just like the other function.

## TODO

There is a paper about using non-linear models for Oaxaca, and variance estimation for two_fold models.

## Authors

* **Austin Adams** - *Initial work* - [AustinJAdams](https://github.com/AustinJAdams)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

* Thank you to Dr. David Slusky for the idea behind the project.

