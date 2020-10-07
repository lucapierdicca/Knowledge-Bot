from numpy.random import binomial
from time import gmtime, strftime

from Q_R.Q_R import Q_R_restore, Q_R_predict, Q_R_model_retrain
from Q_AType.Q_AType import Q_AType_restore, Q_AType_predict, Q_AType_model_retrain
from Q_C1C2_POSTag.Q_C1C2_POS import Q_C1C2_POS_predict, Q_C1C2_POS_restore, Q_C1C2_POS_model_retrain

import willylorbot
import querying
import enriching
import const
import knowledge_server


    
def proba(q_count, e_count):
    normalization = (1.0/q_count) + (1.0/e_count)
    return (1.0/q_count)/normalization


def main():
    

    restore_r = Q_R_restore()
    restore_c1c2 = Q_C1C2_POS_restore()
    restore_Atype= Q_AType_restore()
    
    cursor, conn = const.setting_db()
    
    print('models loaded')
    print('db connection established')


    q_count, e_count = 1, 1
    start_retrain = False
    
    while True:
        if strftime("%H:%M", gmtime()) == '05:00':
            if knowledge_server.retrieve_new_entries(cursor, conn) == 1:
                start_retrain = True

        updates = willylorbot.get_updates(willylorbot.last_update_id) 
        if len(updates["result"]) > 0 and start_retrain==False:
            willylorbot.last_update_id = willylorbot.get_last_update_id(updates) + 1 
            
            recvd_message = updates["result"][0]["message"]["text"]
            chat = updates["result"][0]["message"]["chat"]["id"]
            
            if recvd_message == '/start':
                willylorbot.send_message(const.welcome, chat)
            
            elif recvd_message == 'hey willy':
                if q_count == 1:
                    willylorbot.send_message(const.tell_me, chat)
                    
                    while True:
                        updates = willylorbot.get_updates(willylorbot.last_update_id)
                        if len(updates["result"]) > 0: 
                            willylorbot.last_update_id = willylorbot.get_last_update_id(updates) + 1 
                            break
                    recvd_message = updates["result"][0]["message"]["text"]
                    
                    querying.querying_wf(recvd_message, restore_r, restore_c1c2, restore_Atype, chat, cursor)
                    q_count += 1
                else: 
                    rnd = binomial(1, proba(q_count, e_count))
                    if rnd == 1:
                        willylorbot.send_message(const.tell_me, chat)
                        
                        while True:
                            updates = willylorbot.get_updates(willylorbot.last_update_id)
                            if len(updates["result"]) > 0: 
                                willylorbot.last_update_id = willylorbot.get_last_update_id(updates) + 1 
                                break
                        recvd_message = updates["result"][0]["message"]["text"]
                        querying.querying_wf(recvd_message, restore_r, restore_c1c2, restore_Atype, chat, cursor)
                        q_count += 1
                    else:
                        willylorbot.send_message(const.willy_asks, chat)
                        new_entry = enriching.enriching_wf(recvd_message, restore_r, restore_c1c2, restore_Atype, chat, cursor)
                        r = knowledge_server.send_entry(new_entry)
                        print(r)
                        e_count +=1
            else:
                rnd = binomial(1, proba(q_count, e_count))
                if rnd == 1:
                    willylorbot.send_message(const.magic_words, chat)
                else:
                    willylorbot.send_message(const.sleep, chat)
        
        if start_retrain:
                willylorbot.send_message("I'm studying... do not disturb!", chat)
                
                #retraining the models
                Q_R_model_retrain(restore_r['model'], cursor)
                Q_AType_model_retrain(restore_Atype['model'], cursor)
                Q_C1C2_POS_model_retrain(restore_c1c2['model'], cursor)
                
                willylorbot.send_message("Back online!", chat)
            
                


       

if __name__ == '__main__':
    main()

