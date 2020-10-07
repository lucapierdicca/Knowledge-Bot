import json
import requests
import csv
import const
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer , text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.utils import class_weight
import operator
import pickle
from keras.models import load_model


model_name = 'Q_R.h5'

def Q_R_predict(question, restored):
    pred = restored['model'].predict(pad_sequences(np.asarray(restored['tokenizer'].texts_to_sequences([question])),
                                    maxlen=restored['max_len'], 
                                    padding='post'))
    
    id_to_label = {value:key for key, value in restored['label_to_id'].items()}
    
    return id_to_label[np.argmax(pred)], np.max(pred)

def Q_R_restore():
    tokenizer = pickle.load(open('.\\Q_R\\tokenizer.p', 'rb'))
    label_to_id = pickle.load(open('.\\Q_R\\label_to_id.p','rb'))
    max_len = pickle.load(open('.\\Q_R\\max_len.p','rb'))
    model = load_model('.\\Q_R\\Q_R.h5')
    
    restored = {'tokenizer':tokenizer, 'label_to_id':label_to_id, 'max_len':max_len, 'model':model}
    
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
    	raw_y.append(row[1])
        
    print('Tot. rows: ', len(raw_X))
    
    return raw_X, raw_y


#def data_from_csv():
#    raw_X, raw_y = [],[]
#    f = open('../dataset.txt', 'r', encoding="utf-8")
#    r = csv.reader(f, delimiter='|')
#    
#    for row in r:
#    	raw_X.append(row[0][:-1])
#    	raw_y.append(row[1])
#        
#    f.close()
#    
#    print('Tot. rows: ', len(raw_X))
#    
#    return raw_X, raw_y
    

def preprocess(raw_X, raw_y):

    
    X, y = [],[]
    
    #seleziono solo le domande con un numero di parole € [3...25]
    for index, text in enumerate(raw_X):
        if (len(text_to_word_sequence(text)) >= 3 and len(text_to_word_sequence(text)) <= 25):
            X.append(text)
            y.append(raw_y[index])
            
    print('Tot. selected rows: ', len(X))
    
    label_distr = {}
     
    for i in y:
        if i not in label_distr:
            label_distr[i] = 1
        else:
            label_distr[i] += 1
    
    label_distr = sorted(label_distr.items(), key=operator.itemgetter(1))
    
    print('Label statistic: ', label_distr)
    
    
    #----------------------------------------------
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X)
    int_sequences = tokenizer.texts_to_sequences(X)
    
    pickle.dump(tokenizer, open('tokenizer.p', 'wb')) #offline----------------------------------
    
    
    freq_len ={}
    
    for i in int_sequences:
        if len(i) not in freq_len:
            freq_len[len(i)] = 1
        else:
            freq_len[len(i)] += 1
    
    freq_len = sorted(freq_len.items(), key=operator.itemgetter(0))
    
    print('Q_Len statistic: ', freq_len)
    
    
    max_len = 0
    
    for i in int_sequences:
    	if len(i) > max_len:
    		max_len = len(i)
            
    pickle.dump(max_len, open('max_len.p', 'wb')) #offline---------------------------------------
    
    print('Max_len: ', max_len)
    X = pad_sequences(int_sequences, maxlen=max_len, padding='post')
    
    vocab_size = len(tokenizer.word_index)+1
    
    #-----------------------------------------------------
    
    labels = list(set(y))  
    label_to_id = {}
    
    for index,label in enumerate(labels):
    	label_to_id[label] = index
    
    pickle.dump(label_to_id, open('label_to_id.p', 'wb')) #offline
    
    for key,value in label_to_id.items():
    	print(key, value)
    
    y = [label_to_id[i] for i in y]
    
    
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y), y)
    class_weight_dict = dict(enumerate(class_weights))
    
    y = to_categorical(np.asarray(y))
    
    print(X.shape)
    print(y.shape)
    
    return X, y, tokenizer, vocab_size, class_weight_dict, max_len
    
    #---------------------------------------------type(X_train)----------
def load_embeddings(vocab_size, tokenizer):
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

def Q_R_model_retrain(model, cursor):
    
    raw_X, raw_y = get_data(cursor)
    X, y, _, _, class_weight_dict, _ = preprocess(raw_X, raw_y)
    
    
    epo=1
    batch=128
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    model.fit(X_train, y_train, verbose=1, batch_size=batch, epochs=epo, shuffle=True, class_weight=class_weight_dict)
    
    #saving
    model.save(r"C:\Users\luca_pierdicca\Desktop\Q_R\\" + model_name) #SAVING
    

def Q_R_model_train(X, y, embedding_matrix, class_weight_dict, vocab_size, max_len):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    
    
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers.embeddings import Embedding
    
    class_num = 16
    neurons = 16
    epo=5
    batch=128
    
    
    model = Sequential()
    model.add(Embedding(vocab_size, 50, weights=[embedding_matrix], trainable=False, input_length=max_len, mask_zero=True))
    model.add(LSTM(neurons))
    model.add(Dense(class_num, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'], weighted_metrics=['categorical_crossentropy', 'acc'])
    
    print(model.summary())
    
    history = model.fit(X_train, y_train, validation_data = (X_test, y_test), verbose=1, batch_size=batch, epochs=epo, shuffle=True, class_weight=class_weight_dict)
    
    #saving
    model.save(model_name) #SAVING
    
    
    #computing scores
    y_proba = model.predict(X_test)
    y_pred = np.argmax(y_proba, axis=1)
    
    y_test = np.argmax(y_test, axis=1)
    
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    
    my_metrics = _score(y_test, y_pred)
    
    print(my_metrics)
    
    
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
        


#if __name__ == '__main__':
#    main()