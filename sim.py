from pycorenlp import StanfordCoreNLP
from keras.preprocessing.text import text_to_word_sequence

import numpy as np
from math import sqrt
import operator

#
#from gensim.models import KeyedVectors
#model = KeyedVectors.load_word2vec_format(r"C:\Users\luca_pierdicca\Desktop\glove.6B\glove_w2v_50d.6B.txt", binary=False)
#vector_size = model.vector_size


def pos_noun(sentence):
    sentence_pos_tagged, sentece_word_sequence = [], text_to_word_sequence(sentence)
    nlp = StanfordCoreNLP('http://localhost:9000')
    
    output = nlp.annotate(' '.join(sentece_word_sequence), properties={'annotators': 'pos','outputFormat': 'json'})
    sentence_pos_tagged = [(token['word'], token['pos'], token['index'], len(sentece_word_sequence)) for token in output['sentences'][0]['tokens'] if 'NN' in token['pos']]
    
    return sentence_pos_tagged

def governor_ranking(sentence, question):
    sentece_word_sequence = text_to_word_sequence(sentence)
    
    nlp = StanfordCoreNLP('http://localhost:9000')
    
    output = nlp.annotate(' '.join(sentece_word_sequence), properties={'annotators': 'depparse','outputFormat': 'json'})
    
    rank = {}
    for token in output['sentences'][0]['basicDependencies']:
        if token['governor'] not in rank:
            rank[token['governor']] = 1
        else:
            rank[token['governor']] += 1
    
    rank = [(key, value) for key, value in rank.items()]
    rank = sorted(rank, key=operator.itemgetter(1), reverse=True)
    
    greatest_governor_index = rank[0][0]
    for token in output['sentences'][0]['basicDependencies']:
        if token['dependent'] == greatest_governor_index:
            greatest_governor_gloss = token['dependentGloss']
    
    
    print(output['sentences'][0]['basicDependencies'])
    print(output['sentences'][0]['tokens'])
    for token in output['sentences'][0]['basicDependencies']:
        if token['dependent'] < greatest_governor_index and 'subj' in token['dep']:
            subj_info = (token['dependentGloss'], token['dependent'])
    
    print(subj_info)
            
    for token in output['sentences'][0]['tokens']:
        if subj_info[1] == token['index']:
            if 'NN' in token['pos']  and subj_info[0].lower() not in question.lower():
                subj = subj_info[0]
            else:
                subj = ''
            
    print(subj)    
    
    c2 = []
    if subj == '':
        for token in output['sentences'][0]['tokens']:
            if ('NN' in token['pos'] or 'JJ' in token['pos'] or 'VB' in token['pos'] or 'CD' in token['pos']) and token['index'] > greatest_governor_index:
                c2.append(token['word'])
    else:
        c2.append(subj)
    
    if len(c2)==0:
        c2.append(greatest_governor_gloss)

    return c2
    
   
    
            


def centroid_vector(sentence):
    sentece_word_sequence = text_to_word_sequence(sentence)
        
    tot_sum=np.zeros((vector_size), dtype=np.float)
    for word in sentece_word_sequence:
        if word in model:
            tot_sum += model[word]
        else:
            continue
    
    if tot_sum.nonzero() != vector_size:
        return tot_sum/len(sentece_word_sequence)
    else:
        return -1

def word_vector(word):
    if word in model:
        return model[word]
    else:
        return -1
    
                                        
def cosine_sim(vector_a, vector_b):
    dot_prod = np.dot(vector_a, vector_b)
    magn_a = sqrt(np.dot(vector_a, vector_a))
    magn_b = sqrt(np.dot(vector_b, vector_b))
    
    return dot_prod/(magn_a*magn_b)


    