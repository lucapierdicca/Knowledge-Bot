
from keras.preprocessing.text import Tokenizer , text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from keras.models import load_model
import operator
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import matplotlib.pyplot as plt
from sklearn.utils import class_weight

import const

model_name = 'Q_AType.h5'


def Q_AType_predict(question, restored):
    pred_proba = restored['model'].predict(pad_sequences(np.asarray(restored['tokenizer'].texts_to_sequences([question])),
                                    maxlen=restored['max_len'], 
                                    padding='post'))
    
    #id_to_label = {value:key for key, value in label_to_id.items()}
    pred_label = 0
    if pred_proba >= 0.5:
        pred_label = 1
    
    return pred_label, pred_proba[0][0]
    
    

def Q_AType_restore():
    tokenizer = pickle.load(open('.\\Q_AType\\tokenizer.p', 'rb'))
    label_to_id = pickle.load(open('.\\Q_AType\\label_to_id.p','rb'))
    max_len = pickle.load(open('.\\Q_AType\\max_len.p','rb'))
    model = load_model('.\\Q_AType\\Q_AType.h5')
    
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
    
    raw_X, raw_y, raw_ans = [],[], []

    for row in data:
        raw_X.append(row[0][:-1])
        raw_ans.append(row[2])
        raw_y.append([row[3], row[4]])
        
    print('Tot. rows: ', len(raw_X))
    
    return raw_X, raw_y, raw_ans

def preprocess(raw_X, raw_y, raw_ans):
   
    X, y, ans = [],[],[]

        
    #seleziono solo le domande con un numero di parole € [3...25]
    for index, text in enumerate(raw_X):
        if (len(text_to_word_sequence(text)) >= 3 and len(text_to_word_sequence(text)) <= 25):
            X.append(text)
            ans.append(raw_ans[index])
            y.append(raw_y[index])
            
            
    #-------------------------------------------------------------
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X)
    int_sequences = tokenizer.texts_to_sequences(X)
    
    pickle.dump(tokenizer, open('tokenizer.p', 'wb')) #offline----------------------------
    
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
    
    print('Max_len: ', max_len)
    pickle.dump(max_len, open('max_len.p', 'wb')) #offline-----------------------------------
            
    X_p = pad_sequences(int_sequences, maxlen=max_len, padding='post')
    
    vocab_size = len(tokenizer.word_index)+1
            
    #-----------------------------
               
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
    
    
    label_to_id = {'00':0, '01':1}
    
    pickle.dump(label_to_id, open('label_to_id.p', 'wb')) #offline---------------------------
    
    y_temp = []
    for index, answer in enumerate(ans):
        c1_p, c2_p = 0, 0
        
        c1 = w[index][0].lower()
        c2 = w[index][1].lower()
        
        if answer.lower() == c1:
            c1_p=1
        if answer.lower() == c2:
            c2_p=1
    
    #    if c1 in answer.lower() and c2 in answer.lower():
    #        if len(c1) > len(c2):
    #            c1_p=1
    #        else:
    #            c2_p=1
    #    else:
    #        if c1 in answer.lower():
    #            c1_p=1
    #        if c2 in answer.lower():
    #            c2_p=1
    
        if c1_p*c2_p == 1:
            print(c1,c2)
        y_temp.append(label_to_id[str(c1_p)+str(c2_p)])
    
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_temp), y_temp)
    class_weight_dict = dict(enumerate(class_weights))
    
    label_distr = {}
     
    for i in y_temp:
        if i not in label_distr:
            label_distr[i] = 1
        else:
            label_distr[i] += 1
    
    label_distr = sorted(label_distr.items(), key=operator.itemgetter(1))
    
    print('Label statistic: ', label_distr)
    
        
    #y_temp = to_categorical(np.asarray(y_temp))
    
    y_temp = np.asarray(y_temp)
    
    print(X_p.shape)
    print(y_temp.shape)
    
    return X_p, y_temp, tokenizer, vocab_size, class_weight_dict, max_len
    
    
    #----------------------------
    
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
            
    #----------------------------------------------------------
    
    
def Q_AType_model_retrain(model, cursor):
    
    raw_X, raw_y, raw_ans = get_data(cursor)
    X_p, y_temp, _, _, class_weight_dict, _ = preprocess(raw_X, raw_y, raw_ans)
    
    epo=1
    batch=128
    
    X_train, X_test, y_train, y_test = train_test_split(X_p, y_temp, test_size=0.3, random_state=0)
    model.fit(X_train, y_train, verbose=1, batch_size=batch, epochs=epo, shuffle=True, class_weight=class_weight_dict)
    
    #saving
    model.save(r"C:\Users\luca_pierdicca\Desktop\Q_AType\\" + model_name) #SAVING    



def Q_AType_model_train(X_p, y_temp, embedding_matrix, class_weight_dict, vocab_size, max_len):    
    X_train, X_test, y_train, y_test = train_test_split(X_p, y_temp, test_size=0.3, random_state=0)
    
    
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers.embeddings import Embedding
    
    class_num = 1
    neurons = 4
    epo=5
    batch=128
    
    
    model = Sequential()
    model.add(Embedding(vocab_size, 50, weights=[embedding_matrix], trainable=False, input_length=max_len, mask_zero=True))
    model.add(LSTM(neurons))
    model.add(Dense(class_num, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'], weighted_metrics=['binary_crossentropy', 'acc'])
    
    print(model.summary())
    
    history = model.fit(X_train, y_train, validation_data = (X_test, y_test), verbose=1, batch_size=batch, epochs=epo, shuffle=True, class_weight=class_weight_dict)
    
    #saving
    model.save(model_name) #SAVING
    
    
    
    
    
    #computing scores
    y_proba = model.predict(X_test)
    y_pred = [1 if i>=0.5 else 0 for i in y_proba]
    
    #y_test = np.argmax(y_test, axis=1)
    
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
    