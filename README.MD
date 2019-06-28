# Oaxaca.py

This package attempts to port the functionality of the oaxaca command in STATA to Python. The Oaxaca-Blinder, or Blinder-Oaxaca as some call it,
decomposition attempts to explain gaps in means of groups. It uses the linear models of two given regression equations to show what is explained by regression coefficients and known data and
what is unexplained using the current data. There are two types of Oaxaca-Blinder decompositions, the two-fold and the three-fold,
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

The current are three_fold, two_fold, and plot.

The anticipated are var, to calculate variance, cotton_fix, to adjust for differences in group sizes.

### The Oaxaca object

The Oaxaca object is called when you use the code below.

```
model = Oaxaca(data, by, endo)
```
data is required to be either a Numpy array or a Pandas DataFrame. The by value shows which column to bifurcate the date, and the endo value shows what value you wish to explain.
These are the needed data types depending on what type your data is in. 

data type| Numpy Array | Pandas DataFrame |
---| --- | --- |
by type| integer | string |
endo type| integer | string |


### three_fold

This calculates the three_fold decomposition as described in the Jann paper.

```
characteristic, coefficient, interaction, gap = model.three_fold(plot)
```

If plot is set to True, a plot of the three_fold will be returned with the default settings.

This returns four values, all floats: the characteristic effect, the coefficient effect, the interaction effect, and the differences in means.


## two_fold

This calculates the two_fold decomposition as described in the Jann paper.

```
unexplained, explained, gap = model.two_fold(plot)
```

As in the previous, if the plot is set to True, a plot of the two_fold will be returned.

This returns three values, all floats: the unexplained effect, the explained effect, and the gap in means.


## plot

This creates a customizable plot of the effects.

```
model.plot(plt_type, fig_size, xlabel, ylabel, color1, color2, color3, color4)
```

If you wish to just use default settings, feel free to leave the values blank. The default plot is the three_fold decomposition, change it to two_fold by saying plt_type = 2.

## TODO

There is more error checking, and some more features, mainly variance/t-tests, to be fleshed out.

## Authors

* **Austin Adams** - *Initial work* - [AustinJAdams](https://github.com/AustinJAdams)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

* Thank you to Dr. David Slusky for the idea behind the project.

