import os

import numpy as np
from dtaidistance import dtw
from sklearn import metrics
from cobras_ts.cobras_dtw import COBRAS_DTW
from cobras_ts.querier.labelquerier import LabelQuerier
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import DBA as dba


ucr_path = '/Users/ary/Desktop/Thesis/UCR_TS_Archive_2015'
dataset = 'ECG200'
budget = 100
alpha = 0.5
window = 10

# load the data
data = np.loadtxt(os.path.join(ucr_path,dataset,dataset + '_TEST'), delimiter=',')
series = data[:,1:]
labels = data[:,0]

# construct the affinity matrix
dists = dtw.distance_matrix(series, window=int(0.01 * window * series.shape[1]))
dists[dists == np.inf] = 0
dists = dists + dists.T - np.diag(np.diag(dists))
affinities = np.exp(-dists * alpha)

# initialise cobras_dtw with the precomputed affinities
clusterer = COBRAS_DTW(affinities, LabelQuerier(labels), budget)
clustering, intermediate_clusterings, runtimes, ml, cl = clusterer.cluster()

print(metrics.adjusted_rand_score(clustering.construct_cluster_labeling(),labels))


x = np.arange(1,97)


fig = plt.figure()
ax1 = fig.add_subplot(231)
ax1.set_title('DBA')
plt.xlabel('idx')
plt.ylabel('value')

method = 3
cluster_idx = 0

def plotAllCurves(plt):
    superinstances = clustering.clusters[cluster_idx].super_instances
    indices_of_current_cluster = []
    for superinstance in superinstances:
        indices_of_current_cluster += superinstance.indices
    plt.set_title('series ' + 'count=' + str(len(indices_of_current_cluster)))

    color_arr = cm.rainbow(np.linspace(0, 1, len(indices_of_current_cluster)))
    colors = iter(color_arr)

    for index in indices_of_current_cluster:
        cur_color = next(colors)
        y = series[index]
        plt.plot(x, y, color=cur_color)

def plotRepresentativeCurves(plt):
    plt.set_title('representative series ' + 'count=' + str(len(clustering.clusters[cluster_idx].super_instances)))

    color_arr = cm.rainbow(np.linspace(0, 1, len(clustering.clusters[cluster_idx].super_instances)))
    colors = iter(color_arr)
    for super_instance in clustering.clusters[cluster_idx].super_instances:
        cur_color = next(colors)
        plt.plot(x, series[super_instance.representative_idx], color=cur_color)

def plotDSACurves():
    color_arr = cm.rainbow(np.linspace(0, 1, len(clustering.clusters[cluster_idx].super_instances)))
    colors = iter(color_arr)

    for super_instance in clustering.clusters[cluster_idx].super_instances:
        cur_color = next(colors)
        cur_indices = super_instance.indices
        cur_series = list(map(lambda idx: series[idx], cur_indices))
        series_mean = dba.performDBA(cur_series, 10)
        plt.plot(x, series_mean, color=cur_color)

def plotDSA_Variance():
    cur_indices = []
    for super_instance in clustering.clusters[cluster_idx].super_instances:
        cur_indices.extend(super_instance.indices)
    cur_series = list(map(lambda idx: series[idx], cur_indices))

    series_mean, series_dtw_horiz_var, series_dtw_vertic_var, series_vertic_var = dba.performDBA(cur_series, 10)

    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(236)



    ax4.set_title('DTW Horizontal Variance')
    ax5.set_title('DTW Vertical Variance')
    ax6.set_title('Normal Vertical Variance')

    ax1.plot(x, series_mean, color='purple')
    plotAllCurves(ax2)
    plotRepresentativeCurves(ax3)
    # ax2.plot(x, series_vertic_var, color='grey')
    # ax3.plot(x, series_dtw_horiz_var, color='green')
    ax4.plot(x, series_dtw_horiz_var, color='green')
    ax5.plot(x, series_dtw_vertic_var, color='red')
    ax6.plot(x, series_vertic_var, color='grey')

plotDSA_Variance()




plt.show()