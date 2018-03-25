from reader.reader import reader as rd
from learner.relevance_learner import relevance_learner as rl
from learner.clustering import clustering as cluster
from learner.ranker import ranker as rk
import pickle, numpy as np
from keras.models import load_model

data_folder = './data'


reader = rd(data_folder)
#main_dictionary = reader.crate_main_dictionary()
#img_features = reader.read_img_features()
#word_features = reader.read_word_features()
#query_features = reader.create_query_features()
#div_main_dict = reader.create_div_main_dict()
#rank_main_dict = reader.create_rank_main_dict()
#img_terms = reader.create_img_terms()
#query_terms = reader.create_query_terms()

#print query_terms.keys()[0]
#print query_terms[query_terms.keys()[0]]

#print 'end of read'
#print len(query_terms.keys())

#with open('main_dictionary.pickle', 'wb') as main_dictionary_copy:
#   pickle.dump(main_dictionary, main_dictionary_copy, protocol=pickle.HIGHEST_PROTOCOL)
#with open('img_features.pickle', 'wb') as img_features_copy:
#    pickle.dump(img_features, img_features_copy, protocol=pickle.HIGHEST_PROTOCOL)
#with open('word_features.pickle', 'wb') as word_features_copy:
#    pickle.dump(word_features, word_features_copy, protocol=pickle.HIGHEST_PROTOCOL)
#with open('query_features.pickle', 'wb') as query_features_copy:
#    pickle.dump(query_features, query_features_copy, protocol=pickle.HIGHEST_PROTOCOL)
#with open('div_main_dictionary.pickle', 'wb') as div_main_dict_copy:
#    pickle.dump(div_main_dict, div_main_dict_copy, protocol=pickle.HIGHEST_PROTOCOL)
#with open('rank_main_dict.pickle', 'wb') as rank_main_dict_copy:
#    pickle.dump(rank_main_dict, rank_main_dict_copy, protocol=pickle.HIGHEST_PROTOCOL)
#with open('img_terms.pickle', 'wb') as img_terms_copy:
#    pickle.dump(img_terms, img_terms_copy, protocol=pickle.HIGHEST_PROTOCOL)
#with open('query_terms.pickle', 'wb') as query_terms_copy:
#    pickle.dump(query_terms, query_terms_copy, protocol=pickle.HIGHEST_PROTOCOL)

#print "end of save"

'''
print 'reading ... '
with open('main_dictionary.pickle', 'rb') as main_dictionary_copy:
    main_dictionary = pickle.load(main_dictionary_copy)
with open('img_features.pickle', 'rb') as img_features_copy:
    img_features = pickle.load(img_features_copy)
with open('query_features.pickle', 'rb') as query_features_copy:
    query_features = pickle.load(query_features_copy)
with open('img_terms.pickle', 'rb') as img_terms_copy:
    img_terms = pickle.load(img_terms_copy)
with open('query_terms.pickle', 'rb') as query_terms_copy:
    query_terms = pickle.load(query_terms_copy)
print 'done!'

print 'create training dataset ... '
relevance_train_set, relevance_label_set, mappings = reader.create_relevance_train_label(main_dictionary, img_features, query_features, img_terms, query_terms)
print 'done!'


print 'saving ... '
with open('relevance_train_set.pickle', 'wb') as relevance_train_set_copy:
    pickle.dump(relevance_train_set, relevance_train_set_copy, protocol=pickle.HIGHEST_PROTOCOL)
with open('relevance_label_set.pickle', 'wb') as relevance_label_set_copy:
    pickle.dump(relevance_label_set, relevance_label_set_copy, protocol=pickle.HIGHEST_PROTOCOL)
print 'done!'
with open('mappings.pickle', 'wb') as mappings_copy:
    pickle.dump(mappings, mappings_copy, protocol=pickle.HIGHEST_PROTOCOL)
print 'done!'
'''


print 'reading ... '
#with open('relevance_train_set.pickle', 'rb') as relevance_train_set_copy:
#    relevance_train_set = pickle.load(relevance_train_set_copy)
#with open('relevance_label_set.pickle', 'rb') as relevance_label_set_copy:
#    relevance_label_set = pickle.load(relevance_label_set_copy)
with open('div_main_dictionary.pickle', 'rb') as div_main_dictionary_copy:
    div_main_dict = pickle.load(div_main_dictionary_copy)
with open('img_features.pickle', 'rb') as img_features_copy:
    img_features = pickle.load(img_features_copy)
with open('main_dictionary.pickle', 'rb') as main_dictionary_copy:
    main_dictionary = pickle.load(main_dictionary_copy)
#with open('img_features.pickle', 'rb') as img_features_copy:
#    img_features = pickle.load(img_features_copy)
with open('query_features.pickle', 'rb') as query_features_copy:
    query_features = pickle.load(query_features_copy)
#with open('mappings.pickle', 'rb') as mappings_copy:
#    mappings = pickle.load(mappings_copy)
with open('test_mappings.pickle', 'rb') as test_mappings_copy:
    test_mappings = pickle.load(test_mappings_copy)
with open('relevance_test_set.pickle', 'rb') as relevance_test_set_copy:
    relevance_test_set = pickle.load(relevance_test_set_copy)
#with open('test_relevance_label_set.pickle', 'rb') as test_relevance_label_set_copy:
#    test_relevance_label_set = pickle.load(test_relevance_label_set_copy)
#with open('rank_main_dict.pickle', 'rb') as rank_main_dict_copy:
#    rank_main_dict = pickle.load(rank_main_dict_copy)
with open('train_ranking.pickle', 'rb') as train_ranking_copy:
    train_ranking = pickle.load(train_ranking_copy)
#with open('label_ranking.pickle', 'rb') as label_ranking_copy:
#    label_ranking = pickle.load(label_ranking_copy)
#with open('mappings_ranking.pickle', 'rb') as mappings_ranking_copy:
#    mappings_ranking = pickle.load(mappings_ranking_copy)
#with open('sample_weights.pickle', 'rb') as sample_weights_copy:
#    sample_weights = pickle.load(sample_weights_copy)
with open('test_ranking.pickle', 'rb') as test_ranking_copy:
    test_ranking = pickle.load(test_ranking_copy)
with open('test_mappings_ranking.pickle', 'rb') as test_mappings_ranking_copy:
    test_mappings_ranking = pickle.load(test_mappings_ranking_copy)
with open('img_terms.pickle', 'rb') as img_terms_copy:
    img_terms = pickle.load(img_terms_copy)
with open('query_terms.pickle', 'rb') as query_terms_copy:
    query_terms = pickle.load(query_terms_copy)
print 'done!'


#print 'create ranking dataset ... '
#ranker = rk(query_features, img_features)
#tops = ranker.get_tops(main_dictionary, k=300)
#print np.average([len(tops[query]) for query in tops.keys()])
#ranking_dataset = reader.create_ranking_gt(main_dictionary, tops)
#print 'done!'


################################## Relevance
#relevance_learner = rl(relevance_train_set, relevance_label_set, 250)
#model = relevance_learner.train(True)
model = load_model('./saved_models/keras_relevance_trained_model.h5')
#rl.evaluate(model, relevance_train_set, relevance_label_set)
print 'predicting ... '
predictions = rl.predict_by_mappings(model, relevance_test_set, test_mappings)
print '\n done!'


################################# diversity
#cl = cluster(div_main_dict, img_features)
#print 'predicting ... '
#predictions = cl.predict()
#print 'done!'
#print 'evaluating ... '
#print cl.evaluate(predictions)
#print 'done!'


################################# ranking
print 'ranking by greedy  ... '
ranker = rk(query_features, img_features)
tops = ranker.get_tops(predictions)
print 'done!'
print 'evaluating ... '
ranker.evalutae_tops(tops, main_dictionary, div_main_dict)
print 'done!'


##################################### create ranking train set
#print 'creating ranking train set ... '
#selected = ranker.get_relevants(predictions)
#clusters = ranker.get_clusters(selected)
#pred_div_main_dict = ranker.create_pred_div_main_dict(clusters)
#train_ranking, label_ranking, mappings_ranking, sample_weights = ranker.create_rank_train_label(rank_main_dict, predictions, pred_div_main_dict)
#with open('train_ranking.pickle', 'wb') as train_ranking_copy:
#    pickle.dump(train_ranking, train_ranking_copy, protocol=pickle.HIGHEST_PROTOCOL)
#with open('label_ranking.pickle', 'wb') as label_ranking_copy:
#    pickle.dump(label_ranking, label_ranking_copy, protocol=pickle.HIGHEST_PROTOCOL)
#with open('mappings_ranking.pickle', 'wb') as mappings_ranking_copy:
#    pickle.dump(mappings_ranking, mappings_ranking_copy, protocol=pickle.HIGHEST_PROTOCOL)
#with open('sample_weights.pickle', 'wb') as sample_weights_copy:
#    pickle.dump(sample_weights, sample_weights_copy, protocol=pickle.HIGHEST_PROTOCOL)
#print 'done!'


################################### train ranker
print 'ranking by learning ... '
#model = ranker.train(train_ranking, label_ranking, sample_weights, True)
model = load_model('./saved_models/keras_ranker_trained_model.h5')
rank_pred = ranker.predict_by_mappings(model, test_ranking, test_mappings_ranking)
#print 'average returned: ', np.average([len([img for img in rank_pred[query] if rank_pred[query][img] == 1]) for query in rank_pred.keys()])
tops_attention = ranker.get_tops(rank_pred)
tops_ranknet = ranker.get_trained_tops(train_ranking, predictions)
tops_attention_ranknet = ranker.get_trained_tops(train_ranking, rank_pred)
print 'done!'
print 'evaluating Attention ... '
ranker.evalutae_tops(tops_attention, main_dictionary, div_main_dict)
print 'done!'
print 'evaluating RankNet ... '
ranker.evalutae_tops(tops_ranknet, main_dictionary, div_main_dict)
print 'done!'
print 'evaluating Attention-RankNet ... '
ranker.evalutae_tops(tops_attention_ranknet, main_dictionary, div_main_dict)
print 'done!'



############# ranking by relevance model
#new_train_set = []
#ignores = []
#new_mappings = {}
#count = 0
#for i, seq in enumerate(train_ranking):
#    for j, feature in enumerate(seq):
#        if not all(f == 0.0 for f in feature):
#            new_train_set.append(feature)
#            new_mappings[count] = {
#                'query': mappings_ranking[i][j]['query'],
#                'img': mappings_ranking[i][j]['img']
#            }
#            count += 1
#        else:
#          ignores.append(str(i)+'-'+str(j))
#new_label_set = []
#for i, seq in enumerate(label_ranking):
#    for j, label in enumerate(seq):
#        if str(i)+'-'+str(j) not in ignores:
#            new_label_set.append(label)
#relevance_learner = rl(new_train_set, new_label_set, 152)
#model = relevance_learner.train(False)
#rank_pred = rl.predict_by_mappings(model, new_train_set, new_mappings)