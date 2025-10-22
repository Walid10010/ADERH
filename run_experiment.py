from pyod.models.inne import INNE
from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.lof import LOF
from pyod.models.loda import LODA
from pyod.models.deep_svdd import DeepSVDD
#from pyod.models.so_gaal import SO_GAAL
from pyod.models.ecod import ECOD
from pyod.models.ocsvm import OCSVM
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.ocsvm import OCSVM

from sklearn.cluster import DBSCAN
from deepod.models.tabular import RDP, RCA
from ARDEH import ADERH
from sklearn.metrics import roc_auc_score, average_precision_score
import glob
from sklearn.model_selection import StratifiedShuffleSplit




algo_dic =  {'ADERH':ADERH, 'INNE':INNE, 'IForest':IForest,  'LOF':LOF,'DIF':DIF,  'DeepSVDD':DeepSVDD, 'OCSVM':OCSVM, 'ECOD':ECOD, 'LODA':LODA, 'RCA':RCA, 'RDP':RDP  }
import glob
import numpy as np

sss = StratifiedShuffleSplit(n_splits=3, test_size=0.3, random_state=0 )
# algo_dic ={'ADERH_norm':ADERH}
random_seeds = [0, 1, 2, 1000, 10000]

for li in [glob.glob('Classical/*')]:
 for data_name in li:

    #if '9_c' in data_name:continue
    if '.jpg' in data_name:continue
    tt = False
    for algo_name in algo_dic:
        print(data_name)

        if tt: continue
        data = np.load('{}'.format(data_name),
                   allow_pickle=True)
        X, Y = data['X'], data['y']
        from sklearn.preprocessing import MinMaxScaler
        X = MinMaxScaler().fit_transform(X)

        roc_score_value = 0
        ap_score_value = 0
        from  sklearn.preprocessing import MinMaxScaler
        X = MinMaxScaler().fit_transform(X)
        #if X.shape[0] > 20000 or X.shape[0] < 1000:continue
        if algo_name in (['ADERH', 'INNE', 'IForest', 'DIF']):
            for seed in random_seeds:
             for i, (train_index, test_index) in enumerate(sss.split(X, Y)):
                 algo = algo_dic[algo_name](random_state=seed)
                 algo.fit(X[train_index])
                 outlier_score, label = algo.decision_function(X[test_index]), algo.labels_
                 roc_score_value += roc_auc_score(Y[test_index], outlier_score) / (len(random_seeds) * 3)
                 ap_score_value += average_precision_score(Y[test_index], outlier_score) / (len(random_seeds) * 3)


        elif  algo_name in  ['RCA', 'RDP'] :
            for i, (train_index, test_index) in enumerate(sss.split(X, Y)):
                algo = algo_dic[algo_name](device='cpu', verbose=0)
                algo.fit(X[train_index])
                outlier_score, label = algo.decision_function(X[test_index]), algo.predict(X)
                roc_score_value += roc_auc_score(Y[test_index], outlier_score) / 3
                ap_score_value += average_precision_score(Y[test_index], outlier_score) / 3

        elif  algo_name in ['DBSCAN']:
            algo = DBSCAN()
            y_pred = y_prob = algo.fit_predict(X)
            y_pred[y_pred >= 0] = -2
            y_pred[y_pred == -1] = 1
            y_pred[y_pred == -2] = 0
            roc_score_value = roc_auc_score(Y, y_pred)

        elif algo_name in  ['DeepSVDD'] :
            for i, (train_index, test_index) in enumerate(sss.split(X, Y)):
                algo = algo_dic[algo_name](verbose=0)
                algo.fit(X[train_index])
                outlier_score, label = algo.decision_function(X[test_index]), algo.labels_
                roc_score_value += roc_auc_score(Y[test_index], outlier_score) / 3
                ap_score_value += average_precision_score(Y[test_index], outlier_score) /  3



        else:
            for i, (train_index, test_index) in enumerate(sss.split(X, Y)):
             algo = algo_dic[algo_name]()
             algo.fit(X[train_index])
             outlier_score, label = algo.decision_function(X[test_index]), algo.labels_
             roc_score_value += roc_auc_score(Y[test_index], outlier_score)/3
             ap_score_value += average_precision_score(Y[test_index], outlier_score) / 3


             # print('dataset:{}, algo: {}, roc-score: {} '.format(data_name.split('/')[-1].split('.')[0], algo_name,   roc_score_value))
        print('dataset:{}, algo: {}, roc-score: {:.3f}, ap-score: {:.3f}'.format(data_name.split('/')[-1].split('.')[0], algo_name,   roc_score_value, ap_score_value))
        with open('kdd2025_all_repeat.csv'.format(algo_name), 'a') as file:
            file.write(('dataset:{}, algo: {}, roc-score: {:.3f}, ap-score: {:.3f}'.format(data_name.split('/')[-1].split('.')[0], algo_name,   roc_score_value, ap_score_value)) + '\n')
        #     file.write('ROC: {:.3f}'.format(roc_score_value) + '\n')
        #     file.write('AP: {:.3f}'.format(ap_score_value) + '\n')
        #     file.write('F1: {:.3f}'.format(0) + '\n')





