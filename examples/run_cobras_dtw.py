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
print("series shape = " + str(series.shape[1]))
# construct the affinity matrix
dists = dtw.distance_matrix(series, window=int(0.01 * window * series.shape[1]))
dists[dists == np.inf] = 0
dists = dists + dists.T - np.diag(np.diag(dists))
affinities = np.exp(-dists * alpha)

# initialise cobras_dtw with the precomputed affinities
clusterer = COBRAS_DTW(affinities, LabelQuerier(labels), budget)
clustering, intermediate_clusterings, runtimes, ml, cl = clusterer.cluster()



fig = plt.figure()
ax1 = fig.add_subplot(231)
ax2 = fig.add_subplot(232)
ax3 = fig.add_subplot(233)
ax4 = fig.add_subplot(234)
ax5 = fig.add_subplot(235)
ax6 = fig.add_subplot(236)
ax1.set_title('DBA')
ax4.set_title('DTW Horizontal Variance')
ax5.set_title('DTW Vertical Variance')
ax6.set_title('Normal Vertical Variance')

plt.xlabel('idx')
plt.ylabel('value')
x = np.arange(1, 97)

method = 3
cluster_idx = 0


def specialCondition(current_series):
    for i in np.arange(90,len(current_series)):
        if (current_series[i] > 2):
            return True
    return False

def findAllIndicesOfOneCluster(cluster_idx):
    superinstances = clustering.clusters[cluster_idx].super_instances
    indices_of_current_cluster = []
    for superinstance in superinstances:
        indices_of_current_cluster += superinstance.indices
    return indices_of_current_cluster

def findAllSeriesWithIndices(indices):
    target_series = list(map(lambda idx: series[idx], indices))
    return target_series

def findSpecialIndices():
    indices_of_current_cluster = findAllIndicesOfOneCluster(cluster_idx)
    special_indices = []
    for index in indices_of_current_cluster:
        if specialCondition(series[index]):
            special_indices.append(index)
    return special_indices

    # adjust_series_mat          映射的value
    # adjusted_series_weight_mat 往前映射的距离
    # series_mapping_mat         当前映射的index
def findAdjustedSeries(adjusted_series_mat, series_mapping_mat, adjusted_series_weight_mat, range):
    for i in np.arange(0, adjusted_series_mat.shape[0]):
        x = []
        y = []
        for index in np.arange(range[0], range[1]):
            x.append(series_mapping_mat[i, index])


def getColorIter(size):
    color_arr = cm.rainbow(np.linspace(0, 1, size))
    colors = iter(color_arr)
    return colors

def plotMultiCurves(plt, indices, title, showLabel=False):
    plt.set_title(title + ' count=' + str(len(indices)))
    colors = getColorIter(len(indices))
    indices.sort()
    for index in indices:
        cur_color = next(colors)
        y = series[index]
        if showLabel == True:
            plt.plot(x, y, color=cur_color, label="index=" + str(index))
        else:
            plt.plot(x, y, color=cur_color)
    if showLabel == True & len(indices) < 15:
        plt.legend(loc = 9)

def plotAllCurves(plt):
    indices_of_current_cluster = findAllIndicesOfOneCluster(cluster_idx)
    plotMultiCurves(plt, indices_of_current_cluster, 'Series')

def plotRepresentativeCurves(plt):
    representative_indices_of_current_cluster = list(map(lambda superinstance: superinstance.representative_idx, clustering.clusters[cluster_idx].super_instances))
    plotMultiCurves(plt, representative_indices_of_current_cluster, 'Representative Series')


def plotDSACurves():
    colors = getColorIter(len(clustering.clusters[cluster_idx].super_instances))
    for super_instance in clustering.clusters[cluster_idx].super_instances:
        cur_color = next(colors)
        cur_indices = super_instance.indices
        cur_series = list(map(lambda idx: series[idx], cur_indices))
        series_mean = dba.performDBA(cur_series, 10)
        plt.plot(x, series_mean, color=cur_color)

def plotDBA(plt, series_mean, range = None):
    if (range == None):
        plt.plot(x, series_mean, color='purple')
    else:
        plt.plot(x[range[0]: range[1]], series_mean[range[0]: range[1]], color = 'purple')

def plotDTWHorizontalCurve(plt, series_dtw_horiz_var, range = None):
    if (range == None):
        plt.plot(x, series_dtw_horiz_var, color='green')
    else:
        plt.plot(x[range[0] : range[1]], series_dtw_horiz_var[range[0] : range[1]], color='green')


def plotDTWVerticalCurve(plt, series_dtw_vertic_var, range = None):
    if (range == None):
        plt.plot(x, series_dtw_vertic_var, color='red')
    else:
        plt.plot(x[range[0] : range[1]], series_dtw_vertic_var[range[0] : range[1]], color='red')


def plotVerticalCurve(plt, series_vertic_var, range = None):
    if (range == None):
        plt.plot(x, series_vertic_var, color='grey')
    else:
        plt.plot(x[range[0]: range[1]], series_vertic_var[range[0]: range[1]], color='grey')


def plotSpecialCaseOverall(cur_indices):
    cur_series = findAllSeriesWithIndices(cur_indices)
    series_mean, series_dtw_horiz_var, series_dtw_vertic_var, series_vertic_var = dba.performDBA(cur_series, 3)
    plotDBA(ax1, series_mean)
    plotMultiCurves(ax2, cur_indices, 'Series', True)
    plotDTWHorizontalCurve(ax4,series_dtw_horiz_var)
    plotDTWVerticalCurve(ax5, series_dtw_vertic_var)
    plotVerticalCurve(ax6, series_vertic_var)


def plotOverall():
    cur_series = findAllSeriesWithIndices(findAllIndicesOfOneCluster(cluster_idx))
    series_mean, series_dtw_horiz_var, series_dtw_vertic_var, series_vertic_var, adjusted_series_mat, series_mapping_mat, adjusted_series_weight_mat = dba.performDBA(cur_series, 50)

    plotDBA(ax1, series_mean)
    plotAllCurves(ax2)
    plotRepresentativeCurves(ax3)
    plotDTWHorizontalCurve(ax4,series_dtw_horiz_var)
    plotDTWVerticalCurve(ax5, series_dtw_vertic_var)
    plotVerticalCurve(ax6, series_vertic_var)


def plotSelectedSpan(span_tuple):
    cur_series = findAllSeriesWithIndices(findAllIndicesOfOneCluster(cluster_idx))
    series_mean, series_dtw_horiz_var, series_dtw_vertic_var, series_vertic_var, adjusted_series_mat, series_mapping_mat, adjusted_series_weight_mat = dba.performDBA(cur_series, 3)
    plotDBA(ax1, series_mean, span_tuple)

    plotDTWHorizontalCurve(ax4,series_dtw_horiz_var, span_tuple)
    plotDTWVerticalCurve(ax5, series_dtw_vertic_var, span_tuple)
    plotVerticalCurve(ax6, series_vertic_var, span_tuple)

def main():
    # print(metrics.adjusted_rand_score(clustering.construct_cluster_labeling(),labels))


    special_indices = findSpecialIndices()
    print(special_indices)
    # plotSpecialCaseOverall(special_indices)
    # plotSelectedSpan((70, 80))
    plotOverall()
    plt.show()
if __name__ == '__main__':
    main()