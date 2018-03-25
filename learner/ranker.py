import numpy as np, sys, os
from sklearn.cluster import DBSCAN
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Embedding, Input, LSTM, TimeDistributed, Merge
import keras
from learning2rank.learning2rank.rank import RankNet

_SEQUENCE_LENGTH = 300
_BATCH_SIZE = 32
_EPOCH = 5
_NUM_CLASS = 2
_SAVE_DIR = os.path.join(os.getcwd(), 'saved_models')
_MODEL_NAME = 'keras_ranker_trained_model.h5'
_IMG_NUMBER = 32487

class ranker:
    def __init__(self, query_features, img_features, relevance={}, clusters={}):
        self.query_features = query_features
        self.img_features = img_features
        self.relevance = relevance
        self.clusters = clusters
        self.query_number = len(query_features.keys())
        self.feature_size = len(img_features.values()[0]) + len(query_features.values()[0]) + 2

    def get_tops(self, predictions, k=20):
        selected = self.get_relevants(predictions)
        clusters = self.get_clusters(selected)
        tops = {}
        count = 0
        for query in clusters.keys():
            tops[query] = []
            i = 0
            while len(tops[query]) < k and i < len(selected[query]):
                if len(clusters[query][i % len(clusters[query].keys())]) > 0:
                    tops[query].append(clusters[query][i % len(clusters[query].keys())].pop())
                i += 1
            count += 1
            #print count / float(len(clusters.keys()))
        return tops

    def get_relevants(self, predictions):
        selected = {}
        for query in predictions.keys():
            selected[query] = []
            for img in predictions[query].keys():
                if int(predictions[query][img]) == 1:
                    selected[query].append(img)
        return selected

    def get_clusters(self, selected):
        clusters = {}
        for query in selected.keys():
            clusters[query] = {}
            img_features = np.asarray([self.img_features[img] for img in selected[query]])
            labels = DBSCAN(eps=6, min_samples=2).fit_predict(img_features)
            if -1 in labels:
                labels = [l + 1 for l in labels]
            for i, img in enumerate(selected[query]):
                if int(labels[i]) not in clusters[query].keys():
                    clusters[query][int(labels[i])] = []
                clusters[query][int(labels[i])].append(img)
        return clusters

    def create_pred_div_main_dict(self, clusters):
        div_main_dict = {}
        for query in clusters.keys():
            div_main_dict[query] = {}
            for cluster in clusters[query]:
                for img in clusters[query][cluster]:
                    div_main_dict[query][img] = cluster
        return div_main_dict

    def evalutae_tops(self, tops, main_dictionary, div_main_dict):
        accurate = 0
        clusters_recall = 0
        for query in tops.keys():
            clusters = set()
            micro_accurate = 0
            for img in tops[query]:
                if int(float(main_dictionary[query][img])) == 1:
                    micro_accurate += 1
                    if img in div_main_dict[query].keys():
                        clusters.add(div_main_dict[query][img])
            if len(tops[query]) > 0:
                accurate += micro_accurate / float(len(tops[query]))
            clusters_recall += len(clusters) / float(len(set(div_main_dict[query].values())))
        X = np.max([len(tops[key]) for key in tops.keys()])
        #num_queries = len({k: v for k, v in tops.items() if len(v) > 0}.keys())
        num_queries = len(tops.keys())
        #print 'number of queries: ', num_queries
        precision_at_X = accurate / float(num_queries)
        cluster_recall_at_X = clusters_recall / float(num_queries)
        F1_at_X = 2 * precision_at_X * cluster_recall_at_X / float(precision_at_X+cluster_recall_at_X)
        print "Precision at ", X, " : ", precision_at_X
        print "Cluseter Recall at ", X, ' : ', cluster_recall_at_X
        print "F1 at ", X, " : ", F1_at_X

    def create_rank_train_label(self, rank_main_dict, predictions, pred_div_main_dict):
        train = np.zeros((self.query_number, _SEQUENCE_LENGTH, self.feature_size), dtype=float)
        label = np.zeros((self.query_number, _SEQUENCE_LENGTH), dtype=int)
        mappings = {}
        sample_weights = np.ones((self.query_number, _SEQUENCE_LENGTH), dtype=float)
        for i, query in enumerate(rank_main_dict.keys()):
            mappings[i] = {}
            for j, img in enumerate(rank_main_dict[query].keys()):
                train[i][j][:] = self.img_features[img] + self.query_features[query] + [predictions[query][img]] + [pred_div_main_dict[query][img] if int(predictions[query][img]) == 1 else -1]
                label[i][j] = rank_main_dict[query][img]
                mappings[i][j] = {
                    'query': query,
                    'img': img
                }
        for i, seq in enumerate(train):
            for j, feature in enumerate(seq):
                if all(f == 0.0 for f in feature):
                    sample_weights[i][j] = 0.0
                if int(label[i][j]) == 0:
                    sample_weights[i][j] = 0.17
                else:
                    sample_weights[i][j] = 1.0
        return train, label, mappings, sample_weights

    def train(self, train_set, label_set, sample_weights, save_model=True):
        input = Input(shape=(_SEQUENCE_LENGTH, self.feature_size))
        attention = TimeDistributed(Dense(1, activation='sigmoid'))(input)
        x = Merge(mode='mul')([input, attention])
        x = LSTM(256, return_sequences=True)(x)
        x = Dropout(0.5)(x)
        output = Dense(_NUM_CLASS, activation='sigmoid')(x)
        model = Model(inputs=input, outputs=output)

        # model = Sequential()
        # model.add(LSTM(256, input_shape=(_SEQUENCE_LENGTH, self.feature_size), return_sequences=True))
        # model.add(Dropout(0.5))
        # model.add(Dense(_NUM_CLASS, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      sample_weight_mode="temporal",
                      metrics=['accuracy'])

        label_set = [[[1, 0] if int(l) == 0 else [0, 1] for l in seq] for seq in label_set]
        model.fit(train_set, label_set, sample_weight=sample_weights, batch_size=_BATCH_SIZE, epochs=_EPOCH)

        if save_model:
            if not os.path.isdir(_SAVE_DIR):
                os.makedirs(_SAVE_DIR)
            model_path = os.path.join(_SAVE_DIR, _MODEL_NAME)
            model.save(model_path)
            print('Saved trained model at %s ' % model_path)

        return model

    @staticmethod
    def predict(model, test_set):
        predictions = model.predict(test_set)
        labels = np.zeros(shape=(predictions.shape[0], predictions.shape[1]))
        for i, seq in enumerate(predictions):
            for j, pred in enumerate(seq):
                labels[i][j] = np.argmax(pred)
        return labels

    @staticmethod
    def predict_by_mappings(model, test_set, mappings):
        labels = ranker.predict(model, test_set)
        predictions = {}
        for i, seq in enumerate(labels):
            for j, label in enumerate(seq):
                if j < len(mappings[i]):
                    query = mappings[i][j]['query']
                    img = mappings[i][j]['img']
                    if query not in predictions.keys():
                        predictions[query] = {}
                    predictions[query][img] = label
        return predictions

    def get_trained_tops(self, train_set, predictions, k=20):
        X, mappings = self.prepare_prediction_ranking(predictions, train_set)
        RankNetModel = RankNet.RankNet()
        RankNetModel.loading("RankNet.model", X)
        # print RankNetModel.predictTargets(X, 100)
        scores = np.squeeze(RankNetModel.predict(X), axis=1)
        labels = []
        for i, s in enumerate(scores):
            labels.append({
                "index": i,
                "score": s
            })
        sortedLabels = sorted(labels, key=lambda k: -1 * k['score'])
        tops = {}
        for label in sortedLabels:
            query = mappings[label['index']]['query']
            img = mappings[label['index']]['img']
            if query not in tops.keys():
                tops[query] = []
            elif len(tops[query]) < k:
                tops[query].append(img)
        return tops


    def prepare_prediction_ranking(self, predictions, train_set):
        X = np.zeros(shape=(_IMG_NUMBER, self.feature_size), dtype=float)
        mappings = {}
        index = 0
        for i, query in enumerate(predictions.keys()):
            for j, img in enumerate(predictions[query].keys()):
                X[index][:] = train_set[i][j]
                mappings[index] = {
                    'query': query,
                    'img': img
                }
                index += 1
        return X, mappings
