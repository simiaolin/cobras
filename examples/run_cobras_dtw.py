import os

import numpy as np
from dtaidistance import dtw
from cobras_ts.cobras_dtw import COBRAS_DTW
from cobras_ts.querier.labelquerier import LabelQuerier
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import DBA as dba
from matplotlib.patches import Ellipse
import datetime as dt


ucr_path = '/Users/ary/Desktop/Thesis/UCR_TS_Archive_2015'
#dataset = 'ECG200'
dataset = 'Beef'
budget = 100
alpha = 0.5
window = 10
iteration = 1
cluster_idx = 1


# load the data
data = np.loadtxt(os.path.join(ucr_path,dataset,dataset + '_TEST'), delimiter=',')
series = data[:,1:]
labels = data[:,0]

start = 0
end = series.shape[1]

print("series number = " + str(series.shape[0]))
print("series shape = " + str(series.shape[1]))
# construct the affinity matrix
dt1 = dt.datetime.now()
dists = dtw.distance_matrix(series, window=int(0.01 * window * series.shape[1]))
dt2 = dt.datetime.now()
print("distance matrix use time " + str((dt2 - dt1).seconds))
dists[dists == np.inf] = 0
dists = dists + dists.T - np.diag(np.diag(dists))
affinities = np.exp(-dists * alpha)

# initialise cobras_dtw with the precomputed affinities
clusterer = COBRAS_DTW(affinities, LabelQuerier(labels), budget)
clustering, intermediate_clusterings, runtimes, ml, cl = clusterer.cluster()
print("there are " + str(len(clustering.clusters)) + " clusters")


fig = plt.figure()
ax1 = fig.add_subplot(231)
ax2 = fig.add_subplot(232, sharex = ax1, sharey = ax1)
ax3 = fig.add_subplot(233,  sharex = ax1)
ax4 = fig.add_subplot(234, sharex = ax1, sharey = ax1)
ax5 = fig.add_subplot(235, sharex = ax1, sharey = ax1)
ax6 = fig.add_subplot(236, sharex = ax1, sharey = ax1)
ax1.set_title('DBA')
x = np.arange(0, series.shape[1])


def findAllIndicesOfOneCluster(cluster_idx):
    superinstances = clustering.clusters[cluster_idx].super_instances
    indices_of_current_cluster = []
    for superinstance in superinstances:
        indices_of_current_cluster += superinstance.indices
    return indices_of_current_cluster

def findAllSeriesWithIndices(indices):
    target_series = list(map(lambda idx: series[idx], indices))
    return target_series

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


def plotDBA(plt, series_mean, range = None):
    if (range == None):
        plt.plot(x, series_mean, color='purple')
    elif (range[0] + 1 == range[1]):
        plt.scatter(x[range[0]], series_mean[range[0]], color = 'purple')
    else:
        plt.plot(x[range[0]: range[1]], series_mean[range[0]: range[1]], color = 'purple')

def plotRawAverage(plt, cur_series):
    avg = np.divide(np.sum(cur_series, 0), len(cur_series))
    plt.plot(x, avg, color='purple')

def plotDeviationAroundDBA(ax1, series_mean, series_vertic_var, span_tuple = None):
    lower = series_mean - series_vertic_var
    upper = series_mean + series_vertic_var
    real_x = x
    if span_tuple != None:
        real_x = x[span_tuple[0]: span_tuple[1]]
        lower = lower[span_tuple[0]: span_tuple[1]]
        upper = upper[span_tuple[0]: span_tuple[1]]
    ax1.fill_between(real_x, lower, upper, alpha=0.1)

def plotEclipseAroundDBA(ax1, series_mean, series_dtw_horiz_var, series_dtw_vertic_var, span_tuple):
    color_type_num = 10
    color_arr = cm.rainbow(np.linspace(0, 1, color_type_num))
    ells = [Ellipse(xy=(i, series_mean[i]),
                    width=series_dtw_horiz_var[i], height=series_dtw_vertic_var[i], color=color_arr[i%color_type_num])
            for i in np.arange(span_tuple[0], span_tuple[1])
            ]
    for e in ells:
        ax1.add_artist(e)


def plotDTWHorizontalCurve(plt, series_dtw_horiz_var, range = None):
    plotStatisticsCurve(plt, series_dtw_horiz_var, 'green', range)

def plotDTWVerticalCurve(plt, series_dtw_vertic_var, range = None):
    plotStatisticsCurve(plt,series_dtw_vertic_var, 'red', range)

def plotVerticalCurve(plt, series_vertic_var, range = None):
    plotStatisticsCurve(plt,series_vertic_var, 'grey', range)

def plotDTWSpecialVerticalCurve(plt, series_dtw_special_vertic_var, range = None) :
    plotStatisticsCurve(plt, series_dtw_special_vertic_var, 'purple', range)

def plotStatisticsCurve(plt, values, color, title, range = None) :
    plt.set_title(title)
    if (range == None):
        plt.plot(x, values, color=color)
    elif (range[0] + 1 == range[1]):
        plt.scatter(x[range[0]], values[range[0]], color=color)
    else:
        plt.plot(x[range[0]: range[1]], values[range[0]: range[1]], color=color)

    # adjusted_series_weight_mat 往前映射的距离
    # series_mapping_mat         当前映射的index
    # 注意range是[ a, b )
def plotAdjustedSeries(plt, series_mapping_mat, adjusted_series_weight_mat, range, series):
    plt.set_title("count = " + str(len(series)))
    colors = getColorIter(series_mapping_mat.shape[0])
    for series_index in np.arange(0, series_mapping_mat.shape[0]):
        s1, e1 = dba.getStartAndEndMapping(series_mapping_mat, adjusted_series_weight_mat, series_index, range[1] - 1)
        s2, e2 = dba.getStartAndEndMapping(series_mapping_mat, adjusted_series_weight_mat, series_index, range[0])

        x = np.arange(s2, e1)
        y = list(map(lambda  x: series[series_index][x], x))
        if len(x) == 1:
            plt.scatter(x, y, color=next(colors))
        else:
            plt.plot(x, y, color=next(colors))
        expected_range = np.arange(range[0], range[1])
        non_plotted_range = [item for item in expected_range if item not in x]
        for non_plotted_dot in non_plotted_range:
            plt.scatter(non_plotted_dot, series[series_index][non_plotted_dot], color = 'black')


def plotSelectedSpan(span_tuple):
    cur_series = findAllSeriesWithIndices(findAllIndicesOfOneCluster(cluster_idx))

    series_mean, series_dtw_horiz_var, series_dtw_vertic_var, series_vertic_var, adjusted_series_mat, series_mapping_mat, adjusted_series_weight_mat, series_dtw_special_vertic_var = dba.performDBA(cur_series, iteration)

    #plotDBA(ax1, series_mean, span_tuple)
    plotRawAverage(ax1, cur_series)
    #plotDeviationAroundDBA(ax1, series_mean, series_vertic_var, span_tuple)
    #plotEclipseAroundDBA(ax1, series_mean, series_dtw_horiz_var, series_dtw_vertic_var, span_tuple)

    plotAdjustedSeries(ax2, series_mapping_mat, adjusted_series_weight_mat, span_tuple, cur_series)
    plotStatisticsCurves(series_dtw_horiz_var, series_dtw_vertic_var, series_vertic_var, series_dtw_special_vertic_var, span_tuple)


def plotStatisticsCurves(series_dtw_horiz_var, series_dtw_vertic_var, series_vertic_var, series_dtw_special_vertic_var, span_tuple = None):
    plotStatisticsCurve(ax3, series_dtw_horiz_var, 'green', 'dtw_horizontal_standard_deviation', span_tuple)
    plotStatisticsCurve(ax4, series_dtw_vertic_var, 'red', 'dtw_vertical_standard_deviation', span_tuple)
    plotStatisticsCurve(ax5, series_vertic_var, 'grey' , 'vertical_standard_deviation', span_tuple)
    plotStatisticsCurve(ax6, series_dtw_special_vertic_var, 'purple', 'dtw_special_vertical_standard_deviation', span_tuple)


def main():
    # print(metrics.adjusted_rand_score(clustering.construct_cluster_labeling(),labels))

    # special_indices = findSpecialIndices()
    # print(special_indices)
    # plotSpecialCaseOverall(special_indices)
    # plotSpecialSelectedSpan((85,95), special_indices)
    plotSelectedSpan((start, end))
    plt.show()
if __name__ == '__main__':
    main()


# deprecated
def plotRepresentativeCurves(plt):
    representative_indices_of_current_cluster = list(map(lambda superinstance: superinstance.representative_idx,
                                                         clustering.clusters[cluster_idx].super_instances))
    plotMultiCurves(plt, representative_indices_of_current_cluster, 'Representative Series')

# deprecated
def plotSpecialSelectedSpan(span_tuple, cur_indices):
    cur_series = findAllSeriesWithIndices(cur_indices)
    series_mean, series_dtw_horiz_var, series_dtw_vertic_var, series_vertic_var,  adjusted_series_mat, series_mapping_mat, adjusted_series_weight_mat, series_dtw_special_vertic_var = dba.performDBA(cur_series, 3)
    plotDBA(ax1, series_mean, span_tuple)
    plotAdjustedSeries(ax2, series_mapping_mat, adjusted_series_weight_mat, span_tuple, cur_series)
    plotStatisticsCurves(series_dtw_horiz_var, series_dtw_vertic_var, series_vertic_var, series_dtw_special_vertic_var, span_tuple)

#deprecated
def plotAllCurves(plt):
    indices_of_current_cluster = findAllIndicesOfOneCluster(cluster_idx)
    plotMultiCurves(plt, indices_of_current_cluster, 'Series')

#deprecated
def plotOverallQuick():
    cur_series = findAllSeriesWithIndices(findAllIndicesOfOneCluster(cluster_idx))
    series_mean, series_dtw_horiz_var, series_dtw_vertic_var, series_vertic_var, adjusted_series_mat, series_mapping_mat, adjusted_series_weight_mat, series_dtw_special_vertic_var = dba.performDBA(cur_series, 1)

    plotDBA(ax1, series_mean)
    plotAllCurves(ax2)

    plotStatisticsCurves(series_dtw_horiz_var, series_dtw_vertic_var, series_vertic_var, series_dtw_special_vertic_var)

#deprecated
def plotSpecialCaseOverall(cur_indices):
    cur_series = findAllSeriesWithIndices(cur_indices)
    series_mean, series_dtw_horiz_var, series_dtw_vertic_var, series_vertic_var, a, b, c = dba.performDBA(cur_series, 3)
    plotDBA(ax1, series_mean)
    plotMultiCurves(ax2, cur_indices, 'Series', True)
    plotDTWHorizontalCurve(ax4,series_dtw_horiz_var)
    plotDTWVerticalCurve(ax5, series_dtw_vertic_var)
    plotVerticalCurve(ax6, series_vertic_var)


#deprecated
def plotDSACurves():
    colors = getColorIter(len(clustering.clusters[cluster_idx].super_instances))
    for super_instance in clustering.clusters[cluster_idx].super_instances:
        cur_color = next(colors)
        cur_indices = super_instance.indices
        cur_series = list(map(lambda idx: series[idx], cur_indices))
        series_mean = dba.performDBA(cur_series, 10)
        plt.plot(x, series_mean, color=cur_color)

#deprecated
def findSpecialIndices():
    indices_of_current_cluster = findAllIndicesOfOneCluster(cluster_idx)
    special_indices = []
    for index in indices_of_current_cluster:
        if specialCondition(series[index]):
            special_indices.append(index)
    return special_indices

#deprecated
def specialCondition(current_series):
    for i in np.arange(90,len(current_series)):
        if (current_series[i] > 2):
            return True
    return False