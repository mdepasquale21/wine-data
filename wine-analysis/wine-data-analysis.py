import pandas as pd
from sklearn.datasets import load_wine
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression

from pathlib import Path
out_dir = './output'
Path(out_dir).mkdir(parents=True, exist_ok=True)

# DATA IMPORT AND MANIPULATION ##################################################################################################
all_data = load_wine(return_X_y=False)

wine_features = all_data.feature_names
wine_data = all_data.data

wine_df = pd.DataFrame(wine_data, columns = wine_features)

wine_df.info()

print('\nMINIMUM VALUES')
print(wine_df.min())
print('\nMAXIMUM VALUES')
print(wine_df.max())

################################################################################################################################

#Heatmap
plt.subplots(figsize=(13,10))
heat = wine_df.corr()
sns.heatmap(heat)
sns.heatmap(heat, annot = True)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(out_dir+'/Heatmap.png', dpi = 250)
plt.clf()
plt.close()

#Boxplot
df = wine_df[wine_df.flavanoids.isin(wine_df.flavanoids.value_counts().head().index)]
sns.boxplot(x='flavanoids',y='total_phenols',data=df)
plt.savefig(out_dir+'/Boxplots.png', dpi = 250)
plt.clf()
plt.close()

#Histograms ####################################################################################################################

kde_style = {"color": "darkcyan", "lw": 2, "label": "KDE", "alpha": 0.7}
hist_style = {"histtype": "stepfilled", "linewidth": 3, "color":"darkturquoise", "alpha": 0.25}

#alcohol
#wine_df.alcohol.plot.hist(bins=15,range=(11.03,14.83),figsize=(13,10))
sns.distplot(wine_df.alcohol, kde=True, hist=True, rug=False, kde_kws=kde_style, hist_kws=hist_style)
plt.title('Alcohol Histogram')
plt.xlabel('Alcohol')
plt.ylabel('Frequency')
plt.axvline(wine_df.alcohol.mean(), color='cornflowerblue', alpha=0.8, linestyle='dashed', linewidth=2)
plt.savefig(out_dir+'/Histogram-alcohol.png', dpi = 250)
plt.clf()
plt.close()

#total_phenols
#wine_df.total_phenols.plot.hist(bins=15,range=(0.98,3.88),figsize=(13,10))
sns.distplot(wine_df.total_phenols, kde=True, hist=True, rug=False, kde_kws=kde_style, hist_kws=hist_style)
plt.title('Total Phenols Histogram')
plt.xlabel('Total Phenols')
plt.ylabel('Frequency')
plt.axvline(wine_df.total_phenols.mean(), color='cornflowerblue', alpha=0.8, linestyle='dashed', linewidth=2)
plt.savefig(out_dir+'/Histogram-total_phenols.png', dpi = 250)
plt.clf()
plt.close()

#malic_acid
#wine_df.malic_acid.plot.hist(bins=15,range=(0.74,5.80),figsize=(13,10))
sns.distplot(wine_df.malic_acid, kde=True, hist=True, rug=False, kde_kws=kde_style, hist_kws=hist_style)
plt.title('Malic Acid Histogram')
plt.xlabel('Malic Acid')
plt.ylabel('Frequency')
plt.axvline(wine_df.malic_acid.mean(), color='cornflowerblue', alpha=0.8, linestyle='dashed', linewidth=2)
plt.savefig(out_dir+'/Histogram-malic_acid.png', dpi = 250)
plt.clf()
plt.close()

#flavanoids
#wine_df.flavanoids.plot.hist(bins=15,range=(0.34,5.08),figsize=(13,10))
sns.distplot(wine_df.flavanoids, kde=True, hist=True, rug=False, kde_kws=kde_style, hist_kws=hist_style)
plt.title('Flavanoids Histogram')
plt.xlabel('Flavanoids')
plt.ylabel('Frequency')
plt.axvline(wine_df.flavanoids.mean(), color='cornflowerblue', alpha=0.8, linestyle='dashed', linewidth=2)
plt.savefig(out_dir+'/Histogram-flavanoids.png', dpi = 250)
plt.clf()
plt.close()

#nonflavanoid_phenols
#wine_df.nonflavanoid_phenols.plot.hist(bins=15,range=(0.13,0.66),figsize=(13,10))
sns.distplot(wine_df.nonflavanoid_phenols, kde=True, hist=True, rug=False, kde_kws=kde_style, hist_kws=hist_style)
plt.title('Non-Flavanoid Phenols Histogram')
plt.xlabel('Non-Flavanoid Phenols')
plt.ylabel('Frequency')
plt.axvline(wine_df.nonflavanoid_phenols.mean(), color='cornflowerblue', alpha=0.8, linestyle='dashed', linewidth=2)
plt.savefig(out_dir+'/Histogram-nonflavanoid_phenols.png', dpi = 250)
plt.clf()
plt.close()

#Scatterplot and Regression ####################################################################################################

#Scatterplot of flavanoids vs total_phenols
wine_df.plot.scatter(x='flavanoids', y='total_phenols')
#plt.xscale('log')
plt.xlabel('Flavanoids')
plt.ylabel('Total Phenols')
plt.savefig(out_dir+'/scatterplot-flavanoids-total_phenols.png', dpi = 250)
plt.clf()
plt.close()

#Linear regression of flavanoids vs total_phenols
X = wine_df.loc[:, 'flavanoids'].values
y = wine_df.loc[:, 'total_phenols'].values
reg = LinearRegression().fit(X[:, np.newaxis], y) #X is required to be 2d for LinearRegression, that is why np.newaxis is added in this 1d case
print('\nLINEAR REGRESSION BETWEEN FLAVANOIDS AND TOTAL PHENOLS')
score_f_tp = reg.score(X[:, np.newaxis], y)
print('SCORE:\n', score_f_tp)
print('COEFFICIENT:\n', reg.coef_[0]) #reg.coef_ is an array in general, when fitting multidimensional X values
print('INTERCEPT:\n', reg.intercept_)
wine_df.plot.scatter(x='flavanoids', y='total_phenols')
plt.plot(X, reg.predict(X[:, np.newaxis]), 'r')
#plt.plot(X, reg.coef_[0]*X + reg.intercept_, 'k--') is the same as the predicted by predict method of linear regression!

errorbars_suck = True
# I'd like to plot errorbars giving the distance from model prediction, but they suck as they have arrows and the graph's not clear!
if not(errorbars_suck):
    predicted_y = reg.coef_[0]*X + reg.intercept_
    dy = (predicted_y - y) #errors
    upperlimits = []
    lowerlimits = []
    for i in range(0, len(dy)):
        lowerlimits.append(True)
        upperlimits.append(False)
    plt.errorbar(X, y, yerr=dy, fmt='o', capsize=0, ls='None', uplims=upperlimits, lolims=lowerlimits) #plot errors

plt.xlabel('Flavanoids')
plt.ylabel('Total Phenols')
plt.legend(('Linear Fit (R2={:.3f})'.format(score_f_tp), 'Data'), loc='lower right')
plt.savefig(out_dir+'/scatterplot-LINEAR-FIT-flavanoids-total_phenols.png', dpi = 250)
plt.clf()
plt.close()

want_nomalized_plots = True
if want_nomalized_plots:
    from sklearn.preprocessing import StandardScaler
    X = wine_df.loc[:, 'flavanoids'].values
    y = wine_df.loc[:, 'total_phenols'].values
    n_x = StandardScaler().fit_transform(X[:, np.newaxis]) #required to be 2d for LinearRegression, that is why np.newaxis is added in this 1d case
    n_y = StandardScaler().fit_transform(y[:, np.newaxis]) #required to be 2d for LinearRegression, that is why np.newaxis is added in this 1d case
    wine_df = wine_df.assign(normalized_flavanoids = n_x, axis=1)
    wine_df = wine_df.assign(normalized_total_phenols = n_y)
    wine_df.plot.scatter(x='normalized_flavanoids', y='normalized_total_phenols')
    plt.xlabel('Normalized Flavanoids')
    plt.ylabel('Normalized Total Phenols')
    plt.savefig(out_dir+'/scatterplot-NORMALIZED-flavanoids-total_phenols.png', dpi = 250)
    plt.clf()
    reg = LinearRegression().fit(n_x, n_y)
    print('\nLINEAR REGRESSION BETWEEN NORMALIZED FLAVANOIDS AND NORMALIZED TOTAL PHENOLS')
    norm_score_f_tp = reg.score(n_x, n_y)
    print('SCORE:\n', norm_score_f_tp)
    print('COEFFICIENT:\n', reg.coef_[0])
    print('INTERCEPT:\n', reg.intercept_)
    wine_df.plot.scatter(x='normalized_flavanoids', y='normalized_total_phenols')
    plt.plot(n_x, reg.predict(n_x), 'g')
    plt.xlabel('Normalized Flavanoids')
    plt.ylabel('Normalized Total Phenols')
    plt.legend(('Linear Fit (R2={:.3f})'.format(norm_score_f_tp), 'Data'), loc='lower right')
    plt.savefig(out_dir+'/scatterplot-NORMALIZED-LINEAR-FIT-flavanoids-total_phenols.png', dpi = 250)
    plt.clf()
    plt.close()

#Scatterplot of flavanoids vs od280/od315_of_diluted_wines
wine_df.plot.scatter(x='flavanoids', y='od280/od315_of_diluted_wines')
#plt.xscale('log')
plt.xlabel('Flavanoids')
plt.ylabel('OD280/OD315 of Diluted Wines')
plt.savefig(out_dir+'/scatterplot-flavanoids-OD280_OD315-of-diluted-wines.png', dpi = 250)
plt.clf()
plt.close()

#Linear regression of flavanoids vs od280/od315_of_diluted_wines
xx = wine_df.loc[:, 'flavanoids'].values
yy = wine_df.loc[:, 'od280/od315_of_diluted_wines'].values
reg = LinearRegression().fit(xx[:, np.newaxis], yy) #xx is required to be 2d for LinearRegression, that is why np.newaxis is added in this 1d case
print('\nLINEAR REGRESSION BETWEEN FLAVANOIDS AND OD280/OD315 OF DILUTED WINES')
score_stuff = reg.score(xx[:, np.newaxis], yy)
print('SCORE:\n', score_stuff)
print('COEFFICIENT:\n', reg.coef_[0]) #reg.coef_ is an array in general, when fitting multidimensional X values
print('INTERCEPT:\n', reg.intercept_)
wine_df.plot.scatter(x='flavanoids', y='od280/od315_of_diluted_wines')
plt.plot(xx, reg.predict(xx[:, np.newaxis]), 'r')
plt.xlabel('Flavanoids')
plt.ylabel('OD280/OD315 of Diluted Wines')
plt.legend(('Linear Fit (R2={:.3f})'.format(score_stuff), 'Data'), loc='lower right')
plt.savefig(out_dir+'/scatterplot-LINEAR-FIT-flavanoids-OD280_OD315-of-diluted-wines.png', dpi = 250)
plt.clf()
plt.close()
