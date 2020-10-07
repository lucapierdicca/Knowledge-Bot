import sqlite3
import re
import operator
import json
import random
from keras.preprocessing.text import  text_to_word_sequence
from Q_R.Q_R import Q_R_restore, Q_R_predict
from Q_AType.Q_AType import Q_AType_restore, Q_AType_predict
from Q_C1C2_POSTag.Q_C1C2_POS import Q_C1C2_POS_predict, Q_C1C2_POS_restore
import const
import willylorbot



'''
import csv
from sklearn.model_selection import train_test_split

questions, answers = [],[]


f = open('dataset.txt', 'r', encoding="utf-8")
r = csv.reader(f, delimiter='|')

for row in r:
    questions.append(row[0])
    answers.append(row[2])

f.close()


_, test_questions, _, test_answers = train_test_split(questions, answers, test_size=0.3, random_state=0)
_, test_questions, _, test_answers = train_test_split(test_questions, test_answers, test_size=100/len(test_questions))


#print(test_questions[:20])
#print(test_answers[:20])

predicted, failed = [], []

'''



def jaccard(a, b):
    a = text_to_word_sequence(a)
    b = text_to_word_sequence(b)
    
    intersection = set(a).intersection(set(b))
    union = set(a).union(set(b))
    return len(intersection)/len(union)




"""******************************"""
"""-------QUERYING STEPS---------"""
"""
1. -> Input Question
2. Concept prediction
3. Relation prediction
4. -> Input Domain (yes/no)
5. C1 (possible) disambiguation
6. Answer_type prediction
7. Output Answer -> 
"""
"""------------------------------"""




def querying_wf(question, restore_r, restore_c1c2, restore_Atype, chat, cursor):

   
    answer = ''
       
    print(question)
    print('')
    
       
    
    #predizione della relazione e dei concetti
    c1 = ''
    
    pred_C1C2 = Q_C1C2_POS_predict(question, restore_c1c2)
    print(pred_C1C2)
    
    if 'c1' in pred_C1C2:
        predicted_c1 = ' '.join([i[0] for i in pred_C1C2['c1']])
    if 'c2' in pred_C1C2:
        predicted_c2 = ' '.join([i[0] for i in pred_C1C2['c2']])
    
    pred_R = Q_R_predict(question, restore_r)  
    print(pred_R)
    predicted_relation = pred_R[0].lower()
    
    
    #eseguo query per la disambiguazione di c1 sfruttando la conferma del dominio (user) e i domini permessi dalla relazione
    results_join = cursor.execute(const.query_querying_disambiguation_join % predicted_c1).fetchall()
    
    
    
    
    
    
    #gestione dei possibili casi in funzione della cardinalità dei risultati della query principale
    if len(results_join) == 0:
        """----------------------------------------------------------------------------"""
        print('Unfortunately no results in my KB!')
        answer = "I don't know..."
        
        """"""
        willylorbot.willy_answers("Unfortunately no results in my KB!", chat)
        willylorbot.willy_answers("I don't know...'", chat)
        """"""
    
       
        
        
        
    
    elif len(results_join) > 1:
        """--------------------------------------------------------------------------"""
        results_join = [list(record) for record in results_join]
        
        for rec1 in results_join:
            for rec2 in results_join:
                if rec1[0] == rec2[0]:
                    if rec1[1] == None and rec2[1] != None:
                        rec1.append(0)
                    elif rec2[1] == None and rec1[1] != None:
                        rec2.append(0)
        
        
        results_join = [list(record) for record in results_join if record[-1] != 0]
        
        
        possible_senses = [record[0]+'::'+record[1] for record in results_join]
        
        in_operator = ""
        for sense in possible_senses:
            in_operator += "'"+sense+"',"
        in_operator = in_operator[:-1]
        
        results_join_2 = cursor.execute(const.query_sense_predominance % (in_operator, question)).fetchall()
        
        for record in results_join:
            for record_2 in results_join_2:
                if record[0]+'::'+record[1] == record_2[0]:
                    record.append(record_2[1])
        
        for record in results_join:
            if len(record) != 4:
                record.append(0)
                
        results_join = sorted(results_join, key=operator.itemgetter(3), reverse=True)
        
        results_join_dict = {}
        for record in results_join:
            if record[2] not in results_join_dict:
                results_join_dict[record[2]] = [record]
            else:
                results_join_dict[record[2]].append(record)
                
     
        
        #print(json.dumps(results_join_dict, indent=4))
    
        
        #gestione dei possibili sottocasi in funzione dei domini ammissibili
        possible_domains = list(set(const.relation_to_domains[predicted_relation]).intersection(set(list(results_join_dict.keys()))))
        
        if len(possible_domains) > 1:
            domains_view = ''
            for index, d in enumerate(possible_domains):
                domains_view = domains_view+str(index+1)+'. '+d+'\n'
            
            print("What domain are we talking about?")
            print(domains_view)
            
            """"""
            willylorbot.willy_answers("What domain are we talking about?", chat)
            willylorbot.willy_answers(domains_view, chat)
            while True:
                updates = willylorbot.get_updates(willylorbot.last_update_id)
                if len(updates["result"]) > 0: #se qualcuno ha scritto qualcosa (anche più di una cosa ma a me non servirà)
                    willylorbot.last_update_id = willylorbot.get_last_update_id(updates) + 1 
                    break
            user_domain = int(updates["result"][0]["message"]["text"])
            """"""
                     
            #user_domain = possible_domains[int(input())-1]
            if user_domain not in possible_domains:
                if None in results_join_dict:
                    print("I couldn't find any other domain. Do you want me to guess the answer?")
                   
                    """"""
                    willylorbot.willy_answers("I couldn't find any other domain. Do you want me to guess the answer?", chat)
                    while True:
                        updates = willylorbot.get_updates(willylorbot.last_update_id)
                        if len(updates["result"]) > 0: #se qualcuno ha scritto qualcosa (anche più di una cosa ma a me non servirà)
                            willylorbot.last_update_id = willylorbot.get_last_update_id(updates) + 1 
                            break
                    confirm = updates["result"][0]["message"]["text"]
                    """"""
                    
                    #confirm = input()
                    if confirm == 'no':
                        answer = "Ok, see you..."
                        
                        """"""
                        willylorbot.willy_answers("Ok, see you...", chat)
                       
                        """"""
                    else:
                        if results_join_dict[None][0][1] != None:
                            c1 = results_join_dict[None][0][0]+'::'+results_join_dict[None][0][1]
                        else:
                            c1 = results_join_dict[None][0][0]
                else:
        
                    answer = "Stupid stupid Willy..."
                    
                    """"""
                    willylorbot.willy_answers("Stupid stupid Willy...", chat)
                   
                    """"""
                    
                    
            else:
                c1 = results_join_dict[user_domain][0][0]+'::'+results_join_dict[user_domain][0][1]
        
        elif len(possible_domains) == 1:
            print("We are talking about "+ possible_domains[0]+", right?")
            
            
            """"""
            willylorbot.willy_answers("We are talking about "+ possible_domains[0]+", right?", chat)
            while True:
                updates = willylorbot.get_updates(willylorbot.last_update_id)
                if len(updates["result"]) > 0: #se qualcuno ha scritto qualcosa (anche più di una cosa ma a me non servirà)
                    willylorbot.last_update_id = willylorbot.get_last_update_id(updates) + 1 
                    break
            user_binary_answer = updates["result"][0]["message"]["text"]          
            
            """"""
            
            #user_binary_answer = input()
            if user_binary_answer == 'no':
                if None in results_join_dict:
                    print("I couldn't find any other domain. Do you want me to guess the answer?")
                    
                    """"""
                    willylorbot.willy_answers("I couldn't find any other domain. Do you want me to guess the answer?", chat)
                    while True:
                        updates = willylorbot.get_updates(willylorbot.last_update_id)
                        if len(updates["result"]) > 0: #se qualcuno ha scritto qualcosa (anche più di una cosa ma a me non servirà)
                            willylorbot.last_update_id = willylorbot.get_last_update_id(updates) + 1 
                            break
                    confirm = updates["result"][0]["message"]["text"]
                    """"""                   
                    
                    #confirm = input()
                    if confirm == 'no':
                        answer = "Ok, see you..."
                       
                        """"""
                        willylorbot.willy_answers("Ok, see you...", chat)
                       
                        """"""
                        
                    else:
                        if results_join_dict[None][0][1] != None:
                            c1 = results_join_dict[None][0][0]+'::'+results_join_dict[None][0][1]
                        else:
                            c1 = results_join_dict[None][0][0]
                else:
                    answer = "Stupid stupid Willy..."
                    
                    """"""
                    willylorbot.willy_answers("Stupid stupid Willy...", chat)
                   
                    """"""
            else:
                c1 = results_join_dict[possible_domains[0]][0][0]+'::'+results_join_dict[possible_domains[0]][0][1]
        
        elif len(possible_domains) == 0:
            print("Unfortunately no domains, do you want me to guess the answer?")
            
            
            """"""
            willylorbot.willy_answers("Unfortunately no domains, do you want me to guess the answer?", chat)
            while True:
                updates = willylorbot.get_updates(willylorbot.last_update_id)
                if len(updates["result"]) > 0: #se qualcuno ha scritto qualcosa (anche più di una cosa ma a me non servirà)
                    willylorbot.last_update_id = willylorbot.get_last_update_id(updates) + 1 
                    break
            confirm = updates["result"][0]["message"]["text"]         
            """"""
            
            #confirm = input()
            if confirm == 'no':
                answer = "Ok, see you..."
                
                """"""
                willylorbot.willy_answers("Ok, see you...", chat)
                       
                """"""
            else:
                if results_join_dict[None][0][1] != None:
                    c1 = results_join_dict[None][0][0]+'::'+results_join_dict[None][0][1]
                else:
                    c1 = results_join_dict[None][0][0]
    
    
    
    
    
    elif len(results_join) == 1:
        """---------------------------------------------------------------------------"""
        results_join = [list(record) for record in results_join]
        
        results_join_dict = {}
        for record in results_join:
            if record[2] not in results_join_dict:
                results_join_dict[record[2]] = [record]
            else:
                results_join_dict[record[2]].append(record)
        
        print(results_join_dict)   
        
        possible_domains = list(set(const.relation_to_domains[predicted_relation]).intersection(list(set(results_join_dict.keys()))))
    
        if len(possible_domains) == 1:
            print("We are talking about "+ possible_domains[0]+", right?")
            
            
            """"""
            willylorbot.willy_answers("We are talking about "+ possible_domains[0]+", right?", chat)
            while True:
                updates = willylorbot.get_updates(willylorbot.last_update_id)
                if len(updates["result"]) > 0: #se qualcuno ha scritto qualcosa (anche più di una cosa ma a me non servirà)
                    willylorbot.last_update_id = willylorbot.get_last_update_id(updates) + 1 
                    break
            user_binary_answer = updates["result"][0]["message"]["text"]           
            
            """"""
            
            #user_binary_answer = input()
            if user_binary_answer == 'no':
                if None in results_join_dict:
                    print("I couldn't find any other domain. Do you want me to guess the answer?")
                    
                    """"""
                    willylorbot.willy_answers("I couldn't find any other domain. Do you want me to guess the answer?", chat)
                    while True:
                        updates = willylorbot.get_updates(willylorbot.last_update_id)
                        if len(updates["result"]) > 0: #se qualcuno ha scritto qualcosa (anche più di una cosa ma a me non servirà)
                            willylorbot.last_update_id = willylorbot.get_last_update_id(updates) + 1 
                            break
                    confirm = updates["result"][0]["message"]["text"]
                    """"""                   
                    
                    #confirm = input()
                    if confirm == 'no':
                        answer = "Ok, see you..."
                        
                        """"""
                        willylorbot.willy_answers("Ok, see you...", chat)
                               
                        """"""
                    else:
                        if results_join_dict[None][0][1] != None:
                            c1 = results_join_dict[None][0][0]+'::'+results_join_dict[None][0][1]
                        else:
                            c1 = results_join_dict[None][0][0]
                else:
                    answer = "Stupid stupid Willy..."
                    
                    """"""
                    willylorbot.willy_answers("Stupid stupid Willy...", chat)
                   
                    """"""
            else:
                c1 = results_join_dict[possible_domains[0]][0][0]+'::'+results_join_dict[possible_domains[0]][0][1]
        
        elif len(possible_domains) == 0:
            print("Unfortunately no domains, do you want me to guess the answer?")
            
            
            """"""
            willylorbot.willy_answers("Unfortunately no domains, do you want me to guess the answer?", chat)
            while True:
                updates = willylorbot.get_updates(willylorbot.last_update_id)
                if len(updates["result"]) > 0: #se qualcuno ha scritto qualcosa (anche più di una cosa ma a me non servirà)
                    willylorbot.last_update_id = willylorbot.get_last_update_id(updates) + 1 
                    break
            confirm = updates["result"][0]["message"]["text"]           
            """"""
            
            #confirm = input()
            if confirm == 'no':
                answer = "Ok, see you..."
                        
                """"""
                willylorbot.willy_answers("Ok, see you...", chat)
                       
                """"""
            else:
                if results_join_dict[None][0][1] != None:
                    c1 = results_join_dict[None][0][0]+'::'+results_join_dict[None][0][1]
                else:
                    c1 = results_join_dict[None][0][0]
                        
    
        
    #generazione della risposta
    if answer == '':
        
        pred_A = Q_AType_predict(question, restore_Atype)    
        print(pred_A)
        
        query_c1_condition, query_c2_condition = '', ''
        if 'c1' in pred_C1C2:
            query_c1_condition = " and punct(lower(c1)) = '%s'" % c1
        if 'c2' in pred_C1C2:
            query_c2_condition = " and punct(lower(c2)) regexp '^%s:.*'" %  predicted_c2
        
    
        query_relation_condition = " relation = '%s'" % predicted_relation.upper()
        
    
        query_answer_type_condition = ''
        if pred_A[0] == 1:
            query_answer_type_condition = " and (lower(answer) <> 'yes' and lower(answer) <> 'no')"
        else:
            query_answer_type_condition = " and (lower(answer) = 'yes' or lower(answer) = 'no')"
    
        
        
        query = """SELECT c1, c2, answer, relation, context, question FROM kbs WHERE"""+query_relation_condition+query_c1_condition+query_c2_condition+query_answer_type_condition
        
        results = cursor.execute(query).fetchall()
        
        
        stat = {}
        if len(results) > 1:
            for record in results:
                if record[2] not in stat:
                    stat[record[2]] = 1
                else:
                    stat[record[2]] += 1 
            
            stat = sorted(stat.items(), key=operator.itemgetter(1), reverse=True)   
            answer = stat[0][0]
            
            if pred_A[0] == 1:
                results_struct = cursor.execute(const.query_querying_question_struct % predicted_relation).fetchall()
                results_struct = [list(record) for record in results_struct]
                for record in results_struct:
                    record.append(jaccard(question, record[0]))
                results_struct = sorted(results_struct, key=operator.itemgetter(2), reverse = True)
                answer_struct = results_struct[0][1]
                answer = answer_struct.replace('y', stat[0][0])
            
        elif len(results) == 1:
            answer = results[0][2]
            if pred_A[0] == 1:
                results_struct = cursor.execute(const.query_querying_question_struct % predicted_relation).fetchall()
                results_struct = [list(record) for record in results_struct]
                for record in results_struct:
                    record.append(jaccard(question, record[0]))
                results_struct = sorted(results_struct, key=operator.itemgetter(2), reverse = True)
                answer_struct = results_struct[0][1]
                answer = answer_struct.replace('y', results[0][2])
            
        elif len(results) < 1:
            if pred_A[0] == 1:
                answer = "I don't know..."
                willylorbot.willy_answers("I don't know...", chat)
            else:
                answer = "I'd say no..."
                willylorbot.willy_answers("I'd say no...", chat)
    
    
        print('')
        print('Question: ', question)
        print('Answer: ', answer.upper())
        willylorbot.willy_answers(answer.upper(), chat)
        print('')
        print('')
    
    
    
    #conn.close()
    
