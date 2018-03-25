from sklearn.cluster import DBSCAN
import numpy as np

class clustering:
    def __init__(self, div_main_dict, img_features):
        self.div_main_dict = div_main_dict
        self.img_features = img_features

    def predict(self):
        predictions = {}
        count = 0
        for query in self.div_main_dict.keys():
            predictions[query] = {}
            imgs = self.div_main_dict[query].keys()
            img_features = np.asarray([self.img_features[img] for img in imgs])
            labels = DBSCAN(eps=6, min_samples=2).fit_predict(img_features)
            for i, label in enumerate(labels):
                #if int(label) is not -1:
                    #print label
                predictions[query][imgs[i]] = label
            count += 1
            #print count / float(len(self.div_main_dict.keys()))
        return predictions

    def evaluate(self, predictions):
        #accurate = 0
        count = 0
        dist = 0
        true_cluster_length = []
        pred_cluster_length = []
        for query in self.div_main_dict.keys():
            true_cluster_set = set()
            pred_cluster_set = set()
            for img in self.div_main_dict[query].keys():
                true_cluster = int(self.div_main_dict[query][img])
                pred_cluster = int(predictions[query][img])
                true_cluster_set.add(true_cluster)
                pred_cluster_set.add(pred_cluster)
                #if int(self.div_main_dict[query][img]) == int(predictions[query][img]):
                #    accurate += 1
                #count += 1
            print len(true_cluster_set), ' - ', len(pred_cluster_set)
            dist += abs(len(true_cluster_set) - len(pred_cluster_set))
            true_cluster_length.append(len((true_cluster_set)))
            pred_cluster_length.append(len((pred_cluster_set)))
            #if len(true_cluster_set) == len(pred_cluster_set):
            #    accurate += 1
            count += 1
        print 'true cluster average length: ', np.average(np.asarray(true_cluster_length))
        print 'predicated cluster average length: ', np.average(np.asarray(pred_cluster_length))
        print 'average number of cluster difference:',  dist / float(count)
