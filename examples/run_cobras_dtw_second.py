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
from tslearn.barycenters import euclidean_barycenter, dtw_barycenter_averaging, dtw_barycenter_averaging_subgradient, softdtw_barycenter
from tslearn.datasets import CachedDatasets
from queue import Queue

ucr_path = '/Users/ary/Desktop/Thesis/UCR_TS_Archive_2015'
dataset = 'ECG200'
#dataset = 'Beef'
budget = 100
alpha = 0.5
window = 3
iteration = 1
cluster_idx = 0


# load the data
data = np.loadtxt(os.path.join(ucr_path,dataset,dataset + '_TEST'), delimiter=',')
series = data[:,1:]
labels = data[:,0]

start = 0
end = series.shape[1]

print("series cnt = " + str(series.shape[0]))
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
dt3 = dt.datetime.now()
print("cluster use time " + str((dt3 - dt2).seconds))

print("there are " + str(len(clustering.clusters)) + " clusters")



x = np.arange(0, series.shape[1])


def listToDic(ml):
    dict = {}
    for cml in ml:
        if cml[0] in dict:
            dict[cml[0]].append(cml[1])
        else:
            dict[cml[0]] = [cml[1]]
        if cml[1] in dict:
            dict[cml[1]].append(cml[0])
        else:
            dict[cml[1]] = [cml[0]]
    return dict


ml_dict = listToDic(ml)
cl_dict = listToDic(cl)

def getMustLinks(clustering, first_repre_idx, second_repre_idx):
    cluster_idx_left, cluster_idx_right = getClusteringIdxs(clustering, first_repre_idx, second_repre_idx)
    assert (cluster_idx_left == cluster_idx_right)
    must_links_between_first_and_second = findMustLinksBetweeenSuperintances(ml_dict, first_repre_idx, second_repre_idx)
    return must_links_between_first_and_second

def getCannotLinks(clustering, first_repre_idx, second_repre_idx):
    cluster_idx_left, cluster_idx_right = getClusteringIdxs(clustering, first_repre_idx, second_repre_idx)
    assert (cluster_idx_left != cluster_idx_right)
    cannot_link_between_cluster = getCannotLinkBetweenCluster(cluster_idx_left, cluster_idx_right)
    must_links_between_first_and_cluster = findMustLinksBetweeenSuperintances(ml_dict, first_repre_idx, cannot_link_between_cluster[0])
    must_links_between_second_and_cluster = findMustLinksBetweeenSuperintances(ml_dict, cannot_link_between_cluster[1], second_repre_idx)

    return must_links_between_first_and_cluster, must_links_between_second_and_cluster

def getCannotLinkBetweenCluster(cluster_idx_left, cluster_idx_right):
    right_cluster_sps_indices = []
    for sp in clustering.clusters[cluster_idx_right].super_instances:
        right_cluster_sps_indices.append(sp.representative_idx)

    for sp in clustering.clusters[cluster_idx_left].super_instances:
        sp_repre_idx = sp.representative_idx
        if sp_repre_idx in cl_dict:
            for second in cl_dict[sp_repre_idx]:
                if second in right_cluster_sps_indices:
                    return [sp_repre_idx, second]
    return []
def getClusteringIdxs(clustering, first_repre_idx, second_repre_idx):
    found_first_sp = False
    found_second_sp = False
    cluster_idx_first = 0
    cluster_idx_second = 0
    for cluster_idx in range(0, len(clustering.clusters)):
        cluster = clustering.clusters[cluster_idx]
        for sp in cluster.super_instances:
            if found_first_sp == False and sp.representative_idx == first_repre_idx:
                cluster_idx_first = cluster_idx
                found_first_sp = True
            if found_second_sp == False and sp.representative_idx == second_repre_idx:
                cluster_idx_second = cluster_idx
                found_second_sp = True
            if found_first_sp == True and found_second_sp == True:
                break


    # assert (found_first_sp == True and found_second_sp == True)
    return cluster_idx_first, cluster_idx_second

def findMustLinksBetweeenSuperintances(dict, first_repre_idx, second_repre_idx):
    visited_vertex = []
    visited_path = Queue()
    visited_path.put([first_repre_idx])
    visited_vertex.append(first_repre_idx)

    while (not visited_path.empty()):
        current_path = visited_path.get()
        last_of_path = current_path[len(current_path)-1]

        if last_of_path in dict:
            for next_elem in dict[last_of_path]:
                if (not next_elem in visited_vertex):
                    to_be_added_path = current_path + [next_elem]
                    if (next_elem == second_repre_idx):
                        return to_be_added_path
                    else:
                        visited_path.put(to_be_added_path)
                        visited_vertex.append(next_elem)
    return []


mock_1 = clustering.clusters[0].super_instances[15].representative_idx
mock_2 = clustering.clusters[0].super_instances[9].representative_idx
mock_3 = clustering.clusters[1].super_instances[17].representative_idx

must_link = getMustLinks(clustering, mock_1, mock_2)
cannot_link_1, cannot_link_2 = getCannotLinks(clustering, mock_1, mock_3)


def plotMustLink():
    fig, axs = plt.subplots(len(must_link))
    fig.suptitle('must link')
    for plot_idx in range(0, len(must_link)):
        axs[plot_idx].plot(x, series[must_link[plot_idx]])
    plt.show()

def plotCannotLink():
    fig2, axs2 = plt.subplots(len(cannot_link_1) + len(cannot_link_2))
    fig2.suptitle('can not link')

    for plot_idx in range(0, len(cannot_link_1)):
        axs2[plot_idx].plot(x, series[cannot_link_1[plot_idx]], color = 'green')
    for plot_idx in range(0, len(cannot_link_2)):
        axs2[len(cannot_link_1) +  plot_idx].plot(series[cannot_link_2[plot_idx]], color = 'red')
    plt.show()

# plotMustLink()
plotCannotLink()

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


def plotDBA(plt, series_mean, range = None):
    if (range == None):
        plt.plot(x, series_mean, color='purple')
    elif (range[0] + 1 == range[1]):
        plt.scatter(x[range[0]], series_mean[range[0]], color = 'purple')
    else:
        plt.plot(x[range[0]: range[1]], series_mean[range[0]: range[1]], color = 'purple')

def plotEclipseAroundDBA(ax1, series_mean, series_dtw_horiz_var, series_dtw_vertic_var, span_tuple):
    color_type_num = 10
    color_arr = cm.rainbow(np.linspace(0, 1, color_type_num))
    ells = [Ellipse(xy=(i, series_mean[i]),
                    width=series_dtw_horiz_var[i], height=series_dtw_vertic_var[i], color=color_arr[i%color_type_num])
            for i in np.arange(span_tuple[0], span_tuple[1])
            ]
    for e in ells:
        ax1.add_artist(e)

def plotDeviations(plt, values, color, range = None) :
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




def main():
    # print(metrics.adjusted_rand_score(clustering.construct_cluster_labeling(),labels))

    # special_indices = findSpecialIndices()
    # print(special_indices)
    # plotSpecialCaseOverall(special_indices)
    # plotSpecialSelectedSpan((85,95), special_indices)
    if False:
        plotSelectedSpan((start, end), 1)
    # plotSelectedSpan((start, end), 1)
        plt.show()
if __name__ == '__main__':
    main()

#experimental
def plotDeviationAroundDBA(ax1, series_mean, series_vertic_var, span_tuple = None):
    lower = series_mean - series_vertic_var
    upper = series_mean + series_vertic_var
    real_x = x
    if span_tuple != None:
        real_x = x[span_tuple[0]: span_tuple[1]]
        lower = lower[span_tuple[0]: span_tuple[1]]
        upper = upper[span_tuple[0]: span_tuple[1]]
    ax1.fill_between(real_x, lower, upper, alpha=0.1)

#experimental
def plotRawAverage(plt, cur_series):
    avg = np.divide(np.sum(cur_series, 0), len(cur_series))
    plt.plot(x, avg, color='purple')

#experimental
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

