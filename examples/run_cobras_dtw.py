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
ax1 = fig.add_subplot(111)
ax1.set_title('Scatter Plot')
plt.xlabel('X')
plt.ylabel('Y')

method = 3
cluster_idx = 0

#all the curve
if method == 0:
    color_arr = cm.rainbow(np.linspace(0, 1, 100))
    colors = iter(color_arr)
    for super_instance in clustering.clusters[cluster_idx].size:
        cur_indices = super_instance.indices
        cur_color = next(colors)
        for index in cur_indices:
            y = series[index]
            plt.plot(x, y, color=cur_color)

#representative curve
elif method == 1:
    color_arr = cm.rainbow(np.linspace(0, 1, len(clustering.clusters[cluster_idx].super_instances)))
    colors = iter(color_arr)
    for super_instance in clustering.clusters[cluster_idx].super_instances:
        cur_color = next(colors)
        plt.plot(x,series[super_instance.representative_idx], color = cur_color)

#dsa curve
elif method == 2:
    color_arr = cm.rainbow(np.linspace(0, 1, len(clustering.clusters[cluster_idx].super_instances)))
    colors = iter(color_arr)

    for super_instance in clustering.clusters[cluster_idx].super_instances:
        cur_color = next(colors)
        cur_indices = super_instance.indices
        cur_series = list(map(lambda idx: series[idx], cur_indices))
        series_mean = dba.performDBA(cur_series, 10)
        plt.plot(x, series_mean, color = cur_color)
#dsa curve of the cluster
else:
    cur_indices = []
    for super_instance in clustering.clusters[cluster_idx].super_instances:
        cur_indices.extend(super_instance.indices)
    cur_series = list(map(lambda idx: series[idx], cur_indices))
    series_mean, series_variance = dba.performDBA(cur_series, 10)
    plt.plot(x, series_mean, color = 'purple')


plt.show()