import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.rcParams.update({'font.size': 7})

from pathlib import Path

histograms_folder = './histograms'
scatterplots_folder = './scatterplots'
scatterplots_popularity_folder = './scatterplots_popularity'
scatterplots_study = './scatterplots_study'

Path(histograms_folder).mkdir(parents=True, exist_ok=True)
Path(scatterplots_folder).mkdir(parents=True, exist_ok=True)
Path(scatterplots_popularity_folder).mkdir(parents=True, exist_ok=True)
Path(scatterplots_study).mkdir(parents=True, exist_ok=True)

# import data
dataset = pd.read_csv('wine-reviews-score.csv')

# explore data a bit
print('\nDATA EXPLORATION')
print('\nSHAPE')
print(dataset.shape)
print('\nINFO')
dataset.info()
print('\nDESCRIPTION')
print(dataset.describe())
n_rows_head = 10
print('\nFIRST ' + str(n_rows_head) + ' ENTRIES')
print(dataset.head(n_rows_head))

# distribution properties calculated by pandas description, saved here for ease of use
# these account also for 0s in data (wines with no reviews, no score)
mean_n_reviews = 446.33
mean_av_score = 3.64
median_n_reviews = 66.00
median_av_score = 3.70
q3_n_reviews = 274.00
std_n_reviews = 1867.72

# histograms
# histogram style
kde_style = {"color": "darkcyan", "lw": 1, "label": "KDE", "alpha": 0.7}
hist_style = {"histtype": "stepfilled", "linewidth": 2, "color": "darkturquoise", "alpha": 0.25}

# histogram for n_reviews
sns.distplot(dataset['REVIEWS'], kde=True, hist=True, rug=False, kde_kws=kde_style, hist_kws=hist_style)
plt.title('Number of reviews histogram')
plt.xlabel('Number of reviews')
plt.ylabel('Frequency')
plt.axvline(mean_n_reviews, color='cornflowerblue', alpha=0.8, linestyle='solid', linewidth=1)
plt.axvline(median_n_reviews, color='cornflowerblue', alpha=0.8, linestyle='dashed', linewidth=1)
plt.legend(('KDE', 'mean', 'median'), loc='upper right', bbox_to_anchor=(1.05, 1.15))
plt.savefig(histograms_folder + '/wine_n_reviews_histogram.png', dpi=250)
plt.clf()

# zoom on histogram for n_reviews
sns.distplot(dataset['REVIEWS'], kde=True, hist=True, rug=False, kde_kws=kde_style, hist_kws=hist_style)
plt.title('Number of reviews histogram')
plt.xlabel('Number of reviews')
plt.ylabel('Frequency')
plt.xlim([0, 10000])
plt.axvline(mean_n_reviews, color='cornflowerblue', alpha=0.8, linestyle='solid', linewidth=1)
plt.axvline(median_n_reviews, color='cornflowerblue', alpha=0.8, linestyle='dashed', linewidth=1)
plt.legend(('KDE', 'mean', 'median'), loc='upper right', bbox_to_anchor=(1.05, 1.15))
plt.savefig(histograms_folder + '/wine_n_reviews_histogram_zoom.png', dpi=250)
plt.clf()

# histogram for average_score
sns.distplot(dataset['SCORE'], kde=True, hist=True, rug=False, kde_kws=kde_style, hist_kws=hist_style)
plt.title('Average wine score histogram')
plt.xlabel('Average score')
plt.ylabel('Frequency')
plt.axvline(mean_av_score, color='cornflowerblue', alpha=0.8, linestyle='solid', linewidth=1)
plt.axvline(median_av_score, color='cornflowerblue', alpha=0.8, linestyle='dashed', linewidth=1)
plt.legend(('KDE', 'mean', 'median'), loc='upper right', bbox_to_anchor=(1.05, 1.15))
plt.savefig(histograms_folder + '/wine_average_score_histogram.png', dpi=250)
plt.clf()

plt.close()

# scatter plots
# create scatter plot
n_revs = dataset.iloc[:, 1].values
av_scores = dataset.iloc[:, 2].values

reviews_length = len(n_revs)
scores_length = len(av_scores)

# full scatter plot
plt.xlabel('Number of reviews')
plt.ylabel('Average Score')
plt.title('Wine number of reviews vs average score')
plt.scatter(n_revs, av_scores, c='cyan', marker='o', s=5, linewidth=0.4, edgecolor='grey')
plt.plot(n_revs, [mean_av_score] * reviews_length, c='black', linewidth=0.7)
plt.plot(n_revs, [median_av_score] * reviews_length, c='green', linewidth=0.7)
plt.legend(('mean', 'median'), loc='upper right', bbox_to_anchor=(1.05, 1.15))
plt.plot([mean_n_reviews] * scores_length, av_scores, c='black', linewidth=0.7)
plt.plot([median_n_reviews] * scores_length, av_scores, c='green', linewidth=0.7)
plt.tight_layout()
plt.savefig(scatterplots_folder + '/wines_scatterplot.png', dpi=250)
plt.clf()

plt.close()

# zoom on scatter plot
plt.xlabel('Number of reviews')
plt.ylabel('Average Score')
plt.title('Wine number of reviews vs average score')
# limit plot "around" mean values, account for std dev
plt.xlim([0, mean_n_reviews + std_n_reviews])
plt.ylim([0, 5])
plt.scatter(n_revs, av_scores, c='cyan', marker='o', s=5, linewidth=0.4, edgecolor='grey')
plt.plot([mean_n_reviews] * scores_length, av_scores, c='black', linewidth=0.7)
plt.plot([median_n_reviews] * scores_length, av_scores, c='green', linewidth=0.7)
plt.plot([q3_n_reviews] * scores_length, av_scores, c='blue', linewidth=0.7)
plt.legend(('mean', 'median', '3rd quartile'), loc='upper right', bbox_to_anchor=(1.05, 1.15))
plt.plot(n_revs, [mean_av_score] * reviews_length, c='black', linewidth=0.7)
plt.plot(n_revs, [median_av_score] * reviews_length, c='green', linewidth=0.7)
plt.tight_layout()
plt.savefig(scatterplots_folder + '/wines_scatterplot_zoom.png', dpi=250)
plt.clf()

plt.close()


#####################################################################################################################################################
# use a formula to calculate popularity
# https://math.stackexchange.com/questions/942738/algorithm-to-calculate-rating-based-on-multiple-reviews-using-both-review-score


def popularity(s, n, q, w=0.5, max=5.0):
    """
    :param s: score
    :param n: number of reviews
    :param q: an appropriate number that shows the importance given to the quantity of reviews n
    :param w: weight of the score, must be between 0 and 1
              weight of  the number of reviews is (1-w)
              default w=0.5
    :param max: the maximum possible score, e.g. ratings from 0 to 5, from 0 to 10
                default max=5.0
    :return: returns a number (popularity score) that is between 0 and max
    """
    return w * s + max * (1 - w) * (1 - np.exp(-(n / q)))


max_score = 5.0

# q is the most delicate parameter
# must be a quantity which is supposed to be a moderate reasonable value for the number of reviews
# (e.g. median or mean value?)
factor = np.log(2)

q_mean = mean_n_reviews / factor
q_median = median_n_reviews / factor
q3q = q3_n_reviews / factor
high_q = 1000.0 / factor

# use mean n reviews as moderate value
popularities_mean = [popularity(score, reviews, q_mean) for score, reviews in zip(av_scores, n_revs)]
norm_popularities_mean = [pop / max_score for pop in popularities_mean]

# use median n reviews as moderate value
popularities_median = [popularity(score, reviews, q_median) for score, reviews in zip(av_scores, n_revs)]
norm_popularities_median = [pop / max_score for pop in popularities_median]

# use 3q n reviews as moderate value
popularities_3q = [popularity(score, reviews, q3q) for score, reviews in zip(av_scores, n_revs)]
norm_popularities_3q = [pop / max_score for pop in popularities_3q]

# use 1000 n reviews as moderate value
popularities_high = [popularity(score, reviews, high_q) for score, reviews in zip(av_scores, n_revs)]
norm_popularities_high = [pop / max_score for pop in popularities_high]

# scatter plots with color maps linked to norm popularity

# use median n reviews as moderate value
plt.xlabel('Number of reviews')
plt.ylabel('Average Score')
plt.title('Popularity colormap (moderate=median)')
plt.scatter(n_revs, av_scores, c=norm_popularities_median, cmap='jet', marker='o', s=15, linewidth=0.6)
cbar = plt.colorbar()
cbar.set_label('Normalized popularity')
plt.tight_layout()
plt.savefig(scatterplots_popularity_folder + '/wines_scatterplot_pop_median.png', dpi=250)
plt.clf()

# zoom on median
plt.xlabel('Number of reviews')
plt.ylabel('Average Score')
plt.title('Popularity colormap (moderate=median)')
plt.xlim([-1000, 20000])
plt.scatter(n_revs, av_scores, c=norm_popularities_median, cmap='jet', marker='o', s=15, linewidth=0.6)
cbar = plt.colorbar()
cbar.set_label('Normalized popularity')
plt.tight_layout()
plt.savefig(scatterplots_popularity_folder + '/wines_scatterplot_pop_median_zoom.png', dpi=250)
plt.clf()

# use mean n reviews as moderate value
plt.xlabel('Number of reviews')
plt.ylabel('Average Score')
plt.title('Popularity colormap (moderate=mean)')
plt.scatter(n_revs, av_scores, c=norm_popularities_mean, cmap='jet', marker='o', s=15, linewidth=0.6)
cbar = plt.colorbar()
cbar.set_label('Normalized popularity')
plt.tight_layout()
plt.savefig(scatterplots_popularity_folder + '/wines_scatterplot_pop_mean.png', dpi=250)
plt.clf()

# zoom on mean
plt.xlabel('Number of reviews')
plt.ylabel('Average Score')
plt.title('Popularity colormap (moderate=mean)')
plt.xlim([-1000, 20000])
plt.scatter(n_revs, av_scores, c=norm_popularities_mean, cmap='jet', marker='o', s=15, linewidth=0.6)
cbar = plt.colorbar()
cbar.set_label('Normalized popularity')
plt.tight_layout()
plt.savefig(scatterplots_popularity_folder + '/wines_scatterplot_pop_mean_zoom.png', dpi=250)
plt.clf()

# use 3q n reviews as moderate value
plt.xlabel('Number of reviews')
plt.ylabel('Average Score')
plt.title('Popularity colormap (moderate=3rd quartile)')
plt.scatter(n_revs, av_scores, c=norm_popularities_3q, cmap='jet', marker='o', s=15, linewidth=0.6)
cbar = plt.colorbar()
cbar.set_label('Normalized popularity')
plt.tight_layout()
plt.savefig(scatterplots_popularity_folder + '/wines_scatterplot_pop_3q.png', dpi=250)
plt.clf()

# zoom on 3q
plt.xlabel('Number of reviews')
plt.ylabel('Average Score')
plt.title('Popularity colormap (moderate=3rd quartile)')
plt.xlim([-1000, 20000])
plt.scatter(n_revs, av_scores, c=norm_popularities_3q, cmap='jet', marker='o', s=15, linewidth=0.6)
cbar = plt.colorbar()
cbar.set_label('Normalized popularity')
plt.tight_layout()
plt.savefig(scatterplots_popularity_folder + '/wines_scatterplot_pop_3q_zoom.png', dpi=250)
plt.clf()

# use 1000 n reviews as moderate value
plt.xlabel('Number of reviews')
plt.ylabel('Average Score')
plt.title('Popularity colormap (moderate=1000)')
plt.scatter(n_revs, av_scores, c=norm_popularities_high, cmap='jet', marker='o', s=15, linewidth=0.6)
cbar = plt.colorbar()
cbar.set_label('Normalized popularity')
plt.tight_layout()
plt.savefig(scatterplots_popularity_folder + '/wines_scatterplot_pop_high.png', dpi=250)
plt.clf()

# zoom on 1000
plt.xlabel('Number of reviews')
plt.ylabel('Average Score')
plt.title('Popularity colormap (moderate=1000)')
plt.xlim([-1000, 20000])
plt.scatter(n_revs, av_scores, c=norm_popularities_high, cmap='jet', marker='o', s=15, linewidth=0.6)
cbar = plt.colorbar()
cbar.set_label('Normalized popularity')
plt.tight_layout()
plt.savefig(scatterplots_popularity_folder + '/wines_scatterplot_pop_high_zoom.png', dpi=250)
plt.clf()

#####################################################################################################################################################
# study
# it is better to take into account the long right tail of n reviews
# choosing to use the n reviews mean as moderate value is better, given the long right tail towards very high values
# at the same time we do not want to push too far on very high n reviews because they are few
# so we don't choose a higher moderate value!
# the mean seems the right limit, so from now on q is equal to q_mean

# for w = 0.5 already calculated (popularities_mean and norm_popularities_mean)
# more weight on average scores
a_bit_weight_av_scores = 0.6
popularities_bit_av_scores = [popularity(score, reviews, q_mean, a_bit_weight_av_scores) for score, reviews in zip(av_scores, n_revs)]
norm_popularities_bit_av_scores = [pop / max_score for pop in popularities_bit_av_scores]

more_weight_av_scores = 0.7
popularities_more_av_scores = [popularity(score, reviews, q_mean, more_weight_av_scores) for score, reviews in zip(av_scores, n_revs)]
norm_popularities_more_av_scores = [pop / max_score for pop in popularities_more_av_scores]

a_lot_more_weight_av_scores = 0.8
popularities_a_lot_more_av_scores = [popularity(score, reviews, q_mean, a_lot_more_weight_av_scores) for score, reviews in zip(av_scores, n_revs)]
norm_popularities_a_lot_more_av_scores = [pop / max_score for pop in popularities_a_lot_more_av_scores]

# more weight on the number of reviews
more_weight_n_reviews = 0.3
popularities_more_n_reviews = [popularity(score, reviews, q_mean, more_weight_n_reviews) for score, reviews in zip(av_scores, n_revs)]
norm_popularities_more_n_reviews = [pop / max_score for pop in popularities_more_n_reviews]

# redo the plot for w=0.5
plt.xlabel('Number of reviews')
plt.ylabel('Average Score')
plt.title('Popularity colormap (w=0.5)')
plt.xlim([-1000, 20000])
plt.scatter(n_revs, av_scores, c=norm_popularities_mean, cmap='jet', marker='o', s=15, linewidth=0.6)
cbar = plt.colorbar()
cbar.set_label('Normalized popularity')
plt.tight_layout()
plt.savefig(scatterplots_study + '/w0.5.png', dpi=250)
plt.clf()

# plot for w=0.6
plt.xlabel('Number of reviews')
plt.ylabel('Average Score')
plt.title('Popularity colormap (w=0.6)')
plt.xlim([-1000, 20000])
plt.scatter(n_revs, av_scores, c=norm_popularities_bit_av_scores, cmap='jet', marker='o', s=15, linewidth=0.6)
cbar = plt.colorbar()
cbar.set_label('Normalized popularity')
plt.tight_layout()
plt.savefig(scatterplots_study + '/w0.6.png', dpi=250)
plt.clf()

# plot for w=0.7
plt.xlabel('Number of reviews')
plt.ylabel('Average Score')
plt.title('Popularity colormap (w=0.7)')
plt.xlim([-1000, 20000])
plt.scatter(n_revs, av_scores, c=norm_popularities_more_av_scores, cmap='jet', marker='o', s=15, linewidth=0.6)
cbar = plt.colorbar()
cbar.set_label('Normalized popularity')
plt.tight_layout()
plt.savefig(scatterplots_study + '/w0.7.png', dpi=250)
plt.clf()

# plot for w=0.8
plt.xlabel('Number of reviews')
plt.ylabel('Average Score')
plt.title('Popularity colormap (w=0.8)')
plt.xlim([-1000, 20000])
plt.scatter(n_revs, av_scores, c=norm_popularities_a_lot_more_av_scores, cmap='jet', marker='o', s=15, linewidth=0.6)
cbar = plt.colorbar()
cbar.set_label('Normalized popularity')
plt.tight_layout()
plt.savefig(scatterplots_study + '/w0.8.png', dpi=250)
plt.clf()

# plot for w=0.3
plt.xlabel('Number of reviews')
plt.ylabel('Average Score')
plt.title('Popularity colormap (w=0.3)')
plt.xlim([-1000, 20000])
plt.scatter(n_revs, av_scores, c=norm_popularities_more_n_reviews, cmap='jet', marker='o', s=15, linewidth=0.6)
cbar = plt.colorbar()
cbar.set_label('Normalized popularity')
plt.tight_layout()
plt.savefig(scatterplots_study + '/w0.3.png', dpi=250)
plt.clf()

# ZOOM ON DISTRIBUTION OF POPULARITY #####################################################

# redo the plot for w=0.5
plt.xlabel('Number of reviews')
plt.ylabel('Average Score')
plt.title('Popularity colormap (w=0.5)')
plt.xlim([-200, 1000])
plt.scatter(n_revs, av_scores, c=norm_popularities_mean, cmap='jet', marker='o', s=15, linewidth=0.6)
cbar = plt.colorbar()
cbar.set_label('Normalized popularity')
plt.tight_layout()
plt.savefig(scatterplots_study + '/w0.5_zoom.png', dpi=250)
plt.clf()

# plot for w=0.6
plt.xlabel('Number of reviews')
plt.ylabel('Average Score')
plt.title('Popularity colormap (w=0.6)')
plt.xlim([-200, 1000])
plt.scatter(n_revs, av_scores, c=norm_popularities_bit_av_scores, cmap='jet', marker='o', s=15, linewidth=0.6)
cbar = plt.colorbar()
cbar.set_label('Normalized popularity')
plt.tight_layout()
plt.savefig(scatterplots_study + '/w0.6_zoom.png', dpi=250)
plt.clf()

# plot for w=0.7
plt.xlabel('Number of reviews')
plt.ylabel('Average Score')
plt.title('Popularity colormap (w=0.7)')
plt.xlim([-200, 1000])
plt.scatter(n_revs, av_scores, c=norm_popularities_more_av_scores, cmap='jet', marker='o', s=15, linewidth=0.6)
cbar = plt.colorbar()
cbar.set_label('Normalized popularity')
plt.tight_layout()
plt.savefig(scatterplots_study + '/w0.7_zoom.png', dpi=250)
plt.clf()

# plot for w=0.8
plt.xlabel('Number of reviews')
plt.ylabel('Average Score')
plt.title('Popularity colormap (w=0.8)')
plt.xlim([-200, 1000])
plt.scatter(n_revs, av_scores, c=norm_popularities_a_lot_more_av_scores, cmap='jet', marker='o', s=15, linewidth=0.6)
cbar = plt.colorbar()
cbar.set_label('Normalized popularity')
plt.tight_layout()
plt.savefig(scatterplots_study + '/w0.8_zoom.png', dpi=250)
plt.clf()

# plot for w=0.3
plt.xlabel('Number of reviews')
plt.ylabel('Average Score')
plt.title('Popularity colormap (w=0.3)')
plt.xlim([-200, 1000])
plt.scatter(n_revs, av_scores, c=norm_popularities_more_n_reviews, cmap='jet', marker='o', s=15, linewidth=0.6)
cbar = plt.colorbar()
cbar.set_label('Normalized popularity')
plt.tight_layout()
plt.savefig(scatterplots_study + '/w0.3_zoom.png', dpi=250)
plt.clf()

#####################################################################################################################################################
# study exponential factor
reviews_interval = range(10000)

exp_factor_mean = [(1 - np.exp(-(n / q_mean))) for n in reviews_interval]
exp_factor_median = [(1 - np.exp(-(n / q_median))) for n in reviews_interval]
exp_factor_3q = [(1 - np.exp(-(n / q3q))) for n in reviews_interval]
exp_factor_high_q = [(1 - np.exp(-(n / high_q))) for n in reviews_interval]

plt.xlabel('Number of reviews')
plt.ylabel('Exponential Factor')
# plt.title('')
plt.xlim([-1000, 10000])
plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
plt.plot([n for n in reviews_interval], exp_factor_median, c='green')
plt.plot([n for n in reviews_interval], exp_factor_3q, c='blue')
plt.plot([n for n in reviews_interval], exp_factor_mean, c='black')
plt.plot([n for n in reviews_interval], exp_factor_high_q, c='orange')
plt.legend(('q median', 'q 3rd quartile', 'q mean', 'q high'), loc='upper right', bbox_to_anchor=(1.05, 1.15), ncol=4)
plt.tight_layout()
plt.savefig(scatterplots_study + '/exp_factor_reviews.png', dpi=250)
plt.clf()
