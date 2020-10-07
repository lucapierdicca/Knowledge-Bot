import const
import random
import numpy as np
import sqlite3
import sim
import operator
import willylorbot



def enriching_wf(question, restore_r, restore_c1c2, restore_Atype, chat, cursor):
    
    in_db = False
    rnd_indices = random.sample(range(len(const.domains)-1), 5)
    choosable_domain = [const.domains[index] for index in rnd_indices]
    
    print('What do you want me to ask you about?')
    d_str = ''
    for index, d in enumerate(choosable_domain):
        d_str = d_str+str(index+1)+'. '+d+'\n'
    print(d_str)
    
    """"""
    willylorbot.willy_answers("What do you want me to ask you about?", chat)
    willylorbot.willy_answers(d_str, chat)
    while True:
        updates = willylorbot.get_updates(willylorbot.last_update_id)
        if len(updates["result"]) > 0: #se qualcuno ha scritto qualcosa (anche più di una cosa ma a me non servirà)
            willylorbot.last_update_id = willylorbot.get_last_update_id(updates) + 1 
            break
    choosen_domain = choosable_domain[int(updates["result"][0]["message"]["text"])-1]
    """"""
    #choosen_domain = choosable_domain[int(input())-1]
    
    
    #calcola le sampling probas (inv. prop. al numero di presenze)
    results = cursor.execute(const.query_enriching_probas % (choosen_domain)).fetchall()
    
    if len(results) > 0: #potrebbe non essere presente nel db alcun concetto che appartiene a choosen_domain
        
        in_db = True
        normalization = 0.0
        for record in results:
            normalization += 1.0/(record[1]*record[2])
        probas = np.zeros((len(results)), dtype=np.float)
        
        for index, record in enumerate(results):
            probas[index] = (1.0/(record[1]*record[2]))/normalization
        
        #sampling del concetto (synset_id) secondo le sampling probas appena calcolate
        choosen_synset = results[np.random.multinomial(1, probas).argmax()][0]
        
    
        print(choosen_synset)
        
        #seleziona il tokens del synset appena scelto (lo faccio da senses così non ho errori di spelling!)
        results = cursor.execute(const.query_enriching_tokens_retrieve_tokens % (choosen_synset)).fetchall()
        choosen_tokens = results[0][0]
        choosen_c1 = choosen_tokens+'::'+choosen_synset
        
        #context
        results = cursor.execute(const.query_enriching_retrieve_context % choosen_synset).fetchall()
        context = results[0][0]
        
        
        #se esiste, seleziona una relazione in cui non è mai apparso (random), altrimenti seleziona quella in cui è apparso di meno
        results = cursor.execute(const.query_enriching_relation_from_synset % (choosen_synset)).fetchall()
        appeared_relations = [record[0] for record in results]
        unappeared_relations = list(set(const.domain_to_relations[choosen_domain]).difference(set(appeared_relations)))
        unappeared_relations = [rel.upper() for rel in unappeared_relations]
        
        """-----------??????-------------"""
        #unappeared_relations = [] #WARNING: impostarlo a zero attiva l'else che porta a domande più sensate!
        """------------------------------"""
        
        if len(unappeared_relations) != 0:
            choosen_relation = unappeared_relations[random.randint(0, len(unappeared_relations)-1)]
        else:
            print(results)
            choosen_relation = results[0][0]
        
        print(choosen_relation)
    
    else: #lo vado a pescare da domain 
        results = cursor.execute(const.query_enriching_not_in_db % choosen_domain).fetchall()
        rnd_index = random.randint(0, len(results)-1)
        choosen_tokens = results[rnd_index][0]
        choosen_synset = results[rnd_index][1]
        choosen_c1 = choosen_tokens+'::'+choosen_synset
        
        choosen_relation = const.domain_to_relations[choosen_domain][random.randint(0, len(const.domain_to_relations[choosen_domain])-1)]
    
    
    #seleziona (random) una struttura adeguata per la domanda e la genera
    results = cursor.execute(const.query_enriching_question_struct % choosen_relation).fetchall()
    
    
    question_struct = results[random.randint(0, len(results)-1)][0]
    answer_struct = [record[1] for record in results]
    
    question_generated = question_struct.replace('x', choosen_tokens)
    print(question_generated)
    """"""
    willylorbot.send_message(question_generated, chat)
    
    """"""
    if in_db:
        print('All that I know is:')
        print(context)
        
        """"""
        willylorbot.send_message("All that I know is:", chat)
        willylorbot.send_message(context, chat)
    
        """"""
        
    else:
        print("I don't know anything about it...")
        
        """"""
        willylorbot.send_message("All that I know is:", chat)
        """"""
        
    """"""
    while True:
        updates = willylorbot.get_updates(willylorbot.last_update_id)
        if len(updates["result"]) > 0: #se qualcuno ha scritto qualcosa (anche più di una cosa ma a me non servirà)
            willylorbot.last_update_id = willylorbot.get_last_update_id(updates) + 1 
            break
    answer_from_user = updates["result"][0]["message"]["text"]
    """"""

    #risposta dello user
    #answer_from_user = input()
    
    #    centroid_answer_from_user = sim.centroid_vector(answer_from_user)
    #    
    #    answer_ranking = []
    #    for ans in answer_struct:
    #        answer_ranking.append((ans, sim.cosine_sim(centroid_answer_from_user, sim.centroid_vector(ans))))
    #        
    #    answer_ranking = sorted(answer_ranking, key=operator.itemgetter(1), reverse=True)
    
    c2_list = sim.governor_ranking(answer_from_user, question_generated)
    
    if len(c2_list) > 1:
        c2 = ' '.join(c2_list)
    else:
        c2 = c2_list[0]
    
    postprocessed = {'question':question_generated, 'answer':answer_from_user, 'relation':choosen_relation, 'context':'', 'domains':choosen_domain, 'c1':choosen_c1, 'c2':c2}
    
    print(postprocessed)
    
    return postprocessed
    
