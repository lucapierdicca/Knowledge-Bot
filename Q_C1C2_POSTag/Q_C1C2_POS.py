import csv
from keras.preprocessing.text import Tokenizer , text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from keras.models import load_model
import operator
import re
from pycorenlp import StanfordCoreNLP
import const


model_name = 'Q_C1C2_POS.h5'

def Q_C1C2_POS_predict(question, restored):
    q_word_seq = text_to_word_sequence(question)
    
    q_seq = restored['tokenizer'].texts_to_sequences([question])
    q_seq_p = pad_sequences(np.asarray(q_seq),maxlen=restored['max_len'],padding='post')
    
    nlp = StanfordCoreNLP('http://localhost:9000')
    output = nlp.annotate(' '.join(q_word_seq), properties={'annotators': 'pos','outputFormat': 'json'})
    q_POStagged = ' '.join([token['pos'] for token in output['sentences'][0]['tokens']])
    

    q_POStagged_seq = restored['tokenizer_POS'].texts_to_sequences([q_POStagged])
    q_POStagged_p = pad_sequences(np.asarray(q_POStagged_seq),maxlen=restored['max_len'],padding='post')
    q_POStagged_p = q_POStagged_p.reshape(q_POStagged_p.shape[0],q_POStagged_p.shape[1],1)

    pred = restored['model'].predict([q_seq_p, q_POStagged_p])
    

    c1, c2, others = [],[],[]
    for index, word in enumerate(q_word_seq):
        label = np.argmax(pred[0][index])+1
        proba = np.max(pred[0][index])
        if label == 1:
            c1.append((word, proba))
        if label == 2:
            c2.append((word, proba))
        if label == 3:
            others.append((word, proba))
    
    pred_to_word = {}
    
    if len(c1) > 0:
        pred_to_word['c1'] = c1 # ' '.join(c1)
    if len(c2) > 0:
        pred_to_word['c2'] = c2 # ' '.join(c2)   
    if len(others) > 0:
        pred_to_word['o'] = others
    
    return pred_to_word
    
    

def Q_C1C2_POS_restore():
    tokenizer = pickle.load(open('.\\Q_C1C2_POSTag\\tokenizer.p', 'rb'))
    max_len = pickle.load(open('.\\Q_C1C2_POSTag\\max_len.p','rb'))
    model = load_model('.\\Q_C1C2_POSTag\\Q_C1C2_POS.h5')
    tokenizer_POS = pickle.load(open('.\\Q_C1C2_POSTag\\tokenizer_POS.p','rb'))
    
    restored = {'tokenizer':tokenizer, 'tokenizer_POS':tokenizer_POS, 'max_len':max_len, 'model':model}
    
    return restored


def _score(y_true, y_pred):
	metrics = {}
	tp, tn, fp, fn = 0,0,0,0
	n = len(y_true)

	for i in range(n):
		if y_true[i] == y_pred[i]:
			if y_true[i]:
				tp+=1
			else:
				tn+=1
		else:
			if y_true[i]:
				fn+=1
			else:
				fp+=1

	metrics['accuracy'] = float((tp+tn))/n

	if(tp != 0 or fp !=0):
		metrics['precision'] = float(tp)/(tp+fp)
	else:
		metrics['precision'] = 'NaN'
	if(tp != 0 or fn !=0):
		metrics['recall'] = float(tp)/(tp+fn)
	else:
		metrics['recall'] = 'NaN'

	if metrics['precision'] != 'Nan' and metrics['recall'] != 'Nan':
		metrics['f1'] = 2*float(metrics['precision']*metrics['recall'])/(metrics['precision']+metrics['recall'])
	else:
		metrics['f1'] = 'Nan'
	metrics['count'] = [tp,tn,fp,fn]

	return metrics


def get_data(cursor):
    data = const.select_all_from_kbs_and(cursor)
    
    raw_X, raw_y = [],[]

    for row in data:
        raw_X.append(row[0][:-1])
        raw_y.append([row[3], row[4]])
        
    print('Tot. rows: ', len(raw_X))
    
    return raw_X, raw_y



def preprocess(raw_X, raw_y, retrain=0):
    X, y = [],[]
    
    
    POS_label = []
    
    f = open('PosTags.txt', 'r', encoding="utf-8")
    r = csv.reader(f, delimiter='\t')
    
    for row in r:
    	POS_label.append(row[1])
    f.close()
    
    
    #seleziono solo le domande con un numero di parole € [3...25]
    for index, text in enumerate(raw_X):
        if (len(text_to_word_sequence(text)) >= 3 and len(text_to_word_sequence(text)) <= 25):
            X.append(text)
            y.append(raw_y[index])
            
            
    #-------------------------------------------------------------
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X)
    X_seq = tokenizer.texts_to_sequences(X)
    
    pickle.dump(tokenizer, open('tokenizer.p', 'wb'))
    
    freq_len ={}
    
    for i in X_seq:
        if len(i) not in freq_len:
            freq_len[len(i)] = 1
        else:
            freq_len[len(i)] += 1
            
    freq_len = sorted(freq_len.items(), key=operator.itemgetter(0))
            
    print('Q_Len statistic: ', freq_len)
    
    max_len = 0
    
    for i in X_seq:
    	if len(i) > max_len:
    		max_len = len(i)
    
    print('Max_len: ', max_len)
    pickle.dump(max_len, open('max_len.p', 'wb')) #offline
            
    X_p = pad_sequences(X_seq, maxlen=max_len, padding='post')
    
    vocab_size = len(tokenizer.word_index)+1
            
    #-------------------------------------------------------------

    
    X_POSlabel = pickle.load(open('X_POSlabel.p', 'rb'))
    
    
    if retrain == 1:
        POSlabel_to_numid = {POSlabel:numid+1 for numid, POSlabel in enumerate(POS_label)}
        
        X_POSlabel_new = []
        
        nlp = StanfordCoreNLP('http://localhost:9000')
        
        for q in X[len(X)-len(X_POSlabel)+1:]:
            output = nlp.annotate(' '.join(text_to_word_sequence(q)), properties={'annotators': 'pos','outputFormat': 'json'})
            X_POSlabel_new.append(' '.join([token['pos'] for token in output['sentences'][0]['tokens']]))
        X_POSlabel = X_POSlabel + X_POSlabel_new    
    
    
    
    pickle.dump(X_POSlabel, open('X_POSlabel.p', 'wb'))
       
    tokenizer_POS = Tokenizer()
    tokenizer_POS.fit_on_texts(X_POSlabel)
    X_POS_seq = tokenizer_POS.texts_to_sequences(X_POSlabel)
    
    pickle.dump(tokenizer_POS, open('tokenizer_POS.p', 'wb'))
    
    X_POS_seq_p = pad_sequences(X_POS_seq, maxlen=max_len, padding='post')
    
    X_POS_seq_p = X_POS_seq_p.reshape((X_POS_seq_p.shape[0], X_POS_seq_p.shape[1], 1)) #3D
        
    #------------------------------------------------------------
               
    w = []
    for i in y:
        if i[0].find(':') != -1:
            c1 = i[0][:i[0].find(':')]
        else:
            c1 = i[0]
        if i[1].find(':') != -1:
            c2 = i[1][:i[1].find(':')]
        else:
            c2 = i[1]
            
        w.append([c1, c2])
        
    
    label_to_onehot = {'1':[1,0,0], '2':[0,1,0], '3':[0,0,1] ,'0':[0,0,0]}
    y_temp = []
    
    for index, q in enumerate(X):
        copy = ' '.join(text_to_word_sequence(q)).strip(' ')
    
        
        c1 = ' '.join(text_to_word_sequence(w[index][0])).strip(' ')
    
        c1count = c1.count(' ')
        regc1 = "( |^)%s( |$)" % c1
        regex1 = re.compile(regc1)
        
        c2 = ' '.join(text_to_word_sequence(w[index][1])).strip(' ')
    
        c2count = c2.count(' ')
        regc2 = "( |^)%s( |$)" % c2
        regex2 = re.compile(regc2) 
        
        to_sub_c1 = ' 1 '*(c1count+1)
        to_sub_c2 = ' 2 '*(c2count+1)
        
        
        if (c1 in c2) or (c2 in c1):
            if c1count > c2count:
                copy = regex1.sub(to_sub_c1, copy)
                copy = regex2.sub(to_sub_c2, copy)
            else:
                copy = regex2.sub(to_sub_c2, copy)
                copy = regex1.sub(to_sub_c1, copy)    
        else:
                copy = regex1.sub(to_sub_c1, copy)
                copy = regex2.sub(to_sub_c2, copy)
     
    
        copy = copy.strip(' ')
        
        y_temp.append([i if i in label_to_onehot else '3' for i in text_to_word_sequence(copy)])
    
            
            
    samplew = np.zeros((len(y_temp), max_len))
    
    
    for index, i in enumerate(y_temp):
        for pos, j in enumerate(i):
            samplew[index][pos] = 1.0
    
    y_p = []
    for i in range(len(y_temp)):
        y_p.append(y_temp[i]+['0']*(max_len-len(y_temp[i])))
        
    for seq in y_p:
        for j in range(len(seq)):
            seq[j] = label_to_onehot[seq[j]]
    
    y_p = np.array(y_p)
    
    print(X_p.shape)
    print(y_p.shape)
    
    return X_p, y_p, X_POS_seq_p, tokenizer, vocab_size, samplew, y_temp, X, y
    
    #----------------------------
    
def load_embedings(vocab_size, tokenizer):
    embeddings_index = {}
    f = open('../glove.6B/glove.6B.50d.txt', 'r', encoding='utf-8')
    for line in f:
    	values = line.split()
    	word = values[0]
    	coefs = np.asarray(values[1:], dtype='float32')
    	embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))
    
    #se una parola del vocabolario non è presente fra le glove, lascia il suo embeddings a 0
    embedding_matrix = np.zeros((vocab_size, 50))
    for word, i in tokenizer.word_index.items():
    	embedding_vector = embeddings_index.get(word)
    	if embedding_vector is not None:
    		embedding_matrix[i] = embedding_vector
            
    return embedding_matrix
            
    #----------------------------------------------------------
    
    
def Q_C1C2_POS_model_retrain(model, cursor, X_p, X_POS_seq_p, y_p, samplew, y):
    
    raw_X, raw_y = get_data(cursor)
    X_p, y_p, X_POS_seq_p, _, _, samplew, _, _, y = preprocess(raw_X, raw_y)
    
    epo=1
    batch=128
    
    X_train, X_test, y_train, y_test = train_test_split(X_p, y_p, test_size=0.3, random_state=0)
    X_POS_train, X_POS_test, _, _ = train_test_split(X_POS_seq_p, y_p, test_size=0.3, random_state=0)
    train_sample_weights, test_sample_weights, _, _ = train_test_split(samplew, y, test_size=0.3, random_state=0)
    
    model.fit([X_train, X_POS_train], y_train, verbose=1, batch_size=batch, epochs=epo, shuffle=True, sample_weight=train_sample_weights)
    
    #saving
    model.save(r"C:\Users\luca_pierdicca\Desktop\Q_C1C2_POSTag\\" + model_name) #SAVING  
    
    
    


def Q_C1C2_POS_model_train(X_p, y_p, X_POS_seq_p, y_temp, X, y, samplew, max_len, vocab_size, embedding_matrix):
    
    X_train, X_test, y_train, y_test = train_test_split(X_p, y_p, test_size=0.3, random_state=0)
    X_POS_train, X_POS_test, _, _ = train_test_split(X_POS_seq_p, y_p, test_size=0.3, random_state=0)
    
    train_sample_weights, test_sample_weights, _, _ = train_test_split(samplew, y, test_size=0.3, random_state=0)
    
    _,_,y_true_train_np, y_true_test_np = train_test_split(X, y_temp, test_size=0.3, random_state=0)
    
    
    y_true_train_np = [int(j) for i in y_true_train_np for j in i]
    y_true_test_np_flat = [int(j) for i in y_true_test_np for j in i]
    
    #y_train = np.reshape(y_train, (y_train.shape[0],y_train.shape[1],1))
    #y_test = np.reshape(y_test, (y_test.shape[0],y_test.shape[1],1))
    
    
    
    from keras.models import Model
    from keras.layers import Dense, LSTM, TimeDistributed, Bidirectional, Masking, Input, Concatenate
    from keras.layers.embeddings import Embedding
    

    class_num = 3
    neurons = 8
    epo=15
    batch=64
    
    i1 = Input(shape=(max_len,))
    emb = Embedding(vocab_size, 50, weights=[embedding_matrix], trainable=False, input_length=max_len, mask_zero=True)(i1)
    
    i2 = Input(shape=(max_len, 1))
    mask = Masking(mask_value=0.0)(i2)
    dense = Dense(1, input_shape=(max_len, 1))(mask)
    
    
    
    bi = Bidirectional(LSTM(neurons, return_sequences=True), merge_mode='sum')(emb)
    
    concat = Concatenate()([bi, dense])
    
    td = TimeDistributed(Dense(class_num, activation='softmax'))(concat)
    
    model = Model(inputs=[i1, i2], outputs=td)
    
    print(model.summary())
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'], 
                  weighted_metrics=['accuracy'], 
                  sample_weight_mode="temporal")
    

    for i in range(epo):
    #start training and testing
        history = model.fit([X_train, X_POS_train], y_train, verbose=1, batch_size=batch, epochs=1, shuffle=True, sample_weight=train_sample_weights)
        
        y_pred_train_np, y_pred_test_np, per_sentence = [] ,[], [1]*len(y_test)
        #y_pred_train_p = model.predict(X_train)
        y_pred_test_p = model.predict([X_test, X_POS_test])
        
        #y_pred_train_p = y_pred_train_p.reshape((y_pred_train_p.shape[0], y_pred_train_p.shape[1]))
        
        for index, array in enumerate(y_pred_test_p):
            for j in range(np.count_nonzero(test_sample_weights[index], axis=0)):
    
                label = np.argmax(array[j])+1
                
                if label != int(y_true_test_np[index][j]):
                    per_sentence[index] = 0
                y_pred_test_np.append(label)
            

        my_metrics = _score(y_true_test_np_flat, y_pred_test_np)
        print('Test word acc: ', my_metrics)
        print('Test sentence acc: ', per_sentence.count(1)/len(per_sentence))
    
    
    #saving
    model.save(model_name) #SAVING
    
#if __name__ == '__main__':
#    main()    
