import numpy as np
from dtaidistance import dtw
from sklearn import metrics
from cobras_ts.cobras_dtw import COBRAS_DTW
from cobras_ts.querier.labelquerier import LabelQuerier
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import DBA as dba
from matplotlib.patches import Ellipse



fig = plt.figure()
ax1 = fig.add_subplot(111, aspect = 'auto')
ax1.set_title('Scatter Plot')
plt.xlabel('X')
plt.ylabel('Y')


# ax1.add_artist(e1)
# ax1.plot(x, y)

x  =[1,2]
y = [2,1]
ax1.plot(x, y)
e = Ellipse((0.5, 0.5), 0.2, 0.4)
e1 = Ellipse((0.8, 0.5), 0.2, 0.4)
k = plt.gcf().gca()
k.add_artist(e)
plt.show()


