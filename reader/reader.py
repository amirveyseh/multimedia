import os, sys, csv, numpy as np
from operator import add

_WORD_FEATURE_SIZE = 50
_NUMBER_OF_TRAINING_SAMPLES = 32487
_IMAGE_FEATURE_SIZE = 100

class reader:
    def __init__(self, data_folder_path):
        self.data_folder_path = data_folder_path

    def crate_main_dictionary(self, rgt_folder=None, post_fix=" rGT.txt"):
        if rgt_folder is None:
            rgt_folder = self.data_folder_path + "/gt/rGT"
        self.main_dictionary = {}
        for filename in os.listdir(rgt_folder):
            key_name = filename.replace(post_fix, "")
            self.main_dictionary[key_name] = {}
            with open(rgt_folder + "/" + filename) as f:
                content = f.readlines()
                for line in content:
                    values = line.rstrip().split(",")
                    self.main_dictionary[key_name][values[0]] = values[1]
        return self.main_dictionary

    def read_img_features(self):
        img_feature_folder = self.data_folder_path + "/descCNN"
        self.img_features = {}
        counter = 0
        for filename in os.listdir(img_feature_folder):
            with open(img_feature_folder + "/" + filename, 'rb') as csvfile:
                content = csv.reader(csvfile)
                for line in content:
                    self.img_features[line[0]] = line[1:_IMAGE_FEATURE_SIZE+1]
                    counter += 1
                    print counter / float(32487)
        return self.img_features

    def read_word_features(self):
        self.word_features = {}
        counter = 0
        query_words = self.get_query_words()
        with open(self.data_folder_path + "/vector50-1.txt", 'rb') as file:
            content = file.readlines()
            for line in content:
                values = line.rstrip().split(" ")
                word = values[0]
                if(word in query_words):
                    self.word_features[word] = values[1:]
                counter += 1
                print counter / float(len(content))
        return self.word_features

    def get_wiki_words(self):
        words = set()
        counter = 0
        with open(self.data_folder_path + "/vector50-1.txt", 'rb') as file:
            content = file.readlines()
            for line in content:
                values = line.rstrip().split(" ")
                words.add(values[0])
                counter += 1
                print counter / float(len(content))
        return words

    def get_query_words(self):
        main_dictionary = self.crate_main_dictionary()
        queries = main_dictionary.keys()
        words = set()
        for query in queries:
            query_words = query.split("_")
            for query_word in query_words:
                words.add(query_word.lower())
        return words

    def create_query_features(self):
        main_dictionary = self.crate_main_dictionary()
        queries = main_dictionary.keys()
        word_features = self.read_word_features()
        self.query_featrues = {}
        count = 0
        for query in queries:
            words = query.split("_")
            query_feature = np.zeros(_WORD_FEATURE_SIZE)
            for word in words:
                feature = np.asarray(word_features[word.lower()], dtype=float)
                query_feature = np.add(query_feature, feature)
            query_feature = [f / float(len(words)) for f in query_feature]
            self.query_featrues[query] = query_feature
            count += 1
            print count / float(len(queries))
        return self.query_featrues

    def create_relevance_train_label(self, main_dictionary, image_features, query_features, img_terms, query_terms):
        training_set = np.zeros((_NUMBER_OF_TRAINING_SAMPLES, 3 *_WORD_FEATURE_SIZE +_IMAGE_FEATURE_SIZE), dtype=float)
        label_set = np.zeros((_NUMBER_OF_TRAINING_SAMPLES, 1), dtype=int)
        mappings = []
        count = 0
        for query in main_dictionary.keys():
            for img in main_dictionary[query].keys():
                img_term = list(np.zeros(shape=(_WORD_FEATURE_SIZE), dtype=float))
                query_term = np.zeros(shape=(_WORD_FEATURE_SIZE), dtype=float)
                if img in img_terms.keys():
                    img_term = map(float, img_terms[img])
                if query in query_terms.keys():
                    query_term = map(float, query_terms[query])
                query_feature = map(float, query_features[query])
                image_feature = map(float, image_features[img])
                feature = np.asarray(query_feature + image_feature + img_term + query_term)
                training_set[count] = feature
                label_set[count] = int(main_dictionary[query][img])
                mappings.append({
                    'query': query,
                    'img': img
                })
                count += 1
                print count / float(_NUMBER_OF_TRAINING_SAMPLES)
        print mappings[0]
        return training_set, label_set, mappings

    def create_div_main_dict(self):
        dgt_folder = self.data_folder_path + "/gt/dGT"
        self.div_main_dictionary = {}
        for filename in os.listdir(dgt_folder):
            if filename.endswith("dGT.txt"):
                key_name = filename.replace(" dGT.txt", "")
                self.div_main_dictionary[key_name] = {}
                with open(dgt_folder + "/" + filename) as f:
                    content = f.readlines()
                    for line in content:
                        values = line.rstrip().split(",")
                        self.div_main_dictionary[key_name][values[0]] = values[1]
        return self.div_main_dictionary

    def create_ranking_gt(self, main_dictionary, tops):
        save_dir = self.data_folder_path + "/gt/rankGT"
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        for query in main_dictionary.keys():
            with open(save_dir + "/" + query + " rankGT.txt", 'wb') as file:
                lines = []
                for img in main_dictionary[query].keys():
                    label = 0
                    if img in tops[query]:
                        label = 1
                    lines.append(img + "," + str(label) + "\n")
                file.writelines(lines)
                file.close()

    def create_rank_main_dict(self):
        rankGT_folder = self.data_folder_path + "/gt/rankGT"
        return self.crate_main_dictionary(rankGT_folder, post_fix=" rankGT.txt")

    def create_img_terms(self):
        img_terms = {}
        word_features = self.read_word_features()
        with open(self.data_folder_path + "/desctxt/devset_textTermsPerImage.txt") as file:
            content = file.readlines()
            for line in content:
                values = line.split(" ")
                vector = np.zeros(shape=(50), dtype=float)
                count = 0
                for i in range(1, len(values)):
                    word = values[i].replace('"', "")
                    if word in word_features.keys():
                        vector += map(float, word_features[word])
                        count += 1
                if count != 0:
                    vector /= float(count)
                    img_terms[values[0]] = vector
        return img_terms

    def create_query_terms(self):
        query_terms = {}
        word_features = self.read_word_features()
        progress = 0
        with open(self.data_folder_path + "/desctxt/devset_textTermsPerTopic.txt") as file:
            content = file.readlines()
            for line in content:
                values = line.split(" ")
                vector = np.zeros(shape=(50))
                count = 0
                for i in range(1, len(values)):
                    word = values[i].replace('"', "")
                    if word in word_features.keys():
                        vector += map(float, word_features[word])
                        count += 1
                if count != 0:
                    vector /= float(count)
                    query_terms[values[0]] = vector
                progress += 1
                print progress / float(len(content))
        return query_terms





