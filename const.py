import operator
import sqlite3
import re


def setting_db():
#definizione funzioni per db, connessione al db, restore dei modelli
    def regexp(y, x, search=re.search):
        return 1 if search(y, x) else 0
        
    def punct_replace(str):
        sign = set(str).intersection(set("\'\"!#$%&()*+,-./;<=>@[\\]^_`{|}~"))
        if len(sign) >= 1:
            for ch in sign:
                if ch in str:
                    str=str.replace(ch,"")
        return str
    
    conn = sqlite3.connect('kbs')
    conn.create_function('regexp', 2, regexp)
    conn.create_function('punct', 1, punct_replace)
    cursor = conn.cursor()
    
    return conn, cursor




domain_to_relations = {'Transport and travel':['time','shape','purpose','sound','place','generalization','size','how_to_use','specialization','part'],
    'Food and drink':['shape','smell','purpose','similarity','taste','color','generalization','size','specialization','part'],
    'Culture and society':['purpose','similarity','color','generalization','size','specialization','part'],
    'Textile and clothing':['shape','purpose','similarity','color','material','place','generalization','size','how_to_use','specialization','part'],
    'Language and linguistics':['purpose','similarity','sound','generalization','size','specialization','part'],
    'Heraldry, honors, and vexillology':['shape','purpose','similarity','color','generalization','size','specialization','part'],
    'Mathematics':['time', 'shape','purpose','similarity','generalization','size','how_to_use','specialization','part'],
    'Biology':['activity','shape','smell','similarity','taste','color','generalization','size','specialization','part'],
    'History':['similarity','time','place','generalization','size','specialization','part'],
    'Computing':['time','activity','purpose','similarity','place','generalization','size','how_to_use','specialization','part'],
    'Religion, mysticism and mythology':['time','activity','purpose','similarity','generalization','size','specialization','part'],
    'Games and video games':['time','purpose','similarity','sound','color','generalization','size','how_to_use','specialization','part'],
    'Physics and astronomy':['time','activity','shape','similarity','sound','color','place','generalization','size','specialization','part'],
    'Education':['generalization','how_to_use','specialization','part'],
    'Politics and government':['time','activity','purpose','similarity','color','place','generalization','size','specialization','part'],
    'Meteorology':['smell','sound','taste','color','generalization','size','specialization','part'],
    'Law and crime':['activity','purpose','time','place','generalization','size','specialization','part'],
    'Philosophy and psychology':['activity','similarity','time','generalization','size','specialization','part'],
    'Literature and theatre':['time','activity','purpose','similarity','sound','color','generalization','size','how_to_use','specialization','part'],
    'Warfare and defense':['purpose','time','material','place','generalization','size','how_to_use','specialization','part'],
    'Royalty and nobility':['shape','similarity','time','color','generalization','size','specialization','part'],
    'Media':['time','shape','similarity','sound','color','generalization','size','how_to_use','specialization','part'],
    'Geography and places':['place', 'shape','similarity','color','generalization','size','specialization','part'],
    'Art, architecture, and archaeology':['place','shape','time','color','material','generalization','size','how_to_use','specialization','part'],
    'Health and medicine':['time','activity','smell','purpose','taste','generalization','size','specialization','part'],
    'Numismatics and currencies':['time','shape','purpose','similarity','generalization','size','specialization','part'],
    'Sport and recreation':['time','activity','shape','purpose','color','place','generalization','size','how_to_use','specialization','part'],
    'Animals':['activity','shape','smell','purpose','similarity','sound','color','place','generalization','size','specialization','part'],
    'Business, economics, and finance':['activity','purpose','time','generalization','size','how_to_use','specialization','part'],
    'Music':['time','shape','purpose','similarity','sound','generalization','size','how_to_use','specialization','part'],
    'Chemistry and mineralogy':['shape','smell','purpose','similarity','taste','color','generalization','size','specialization','part'],
    'Engineering and technology':['time','shape','purpose','similarity','sound','material','generalization','how_to_use','specialization','part'],
    'Geology and geophysics':['time','shape','similarity','sound','color','generalization','size','specialization','part'],
    'Farming':['activity','smell','purpose','sound','taste','generalization','size','how_to_use','specialization','part']
        }



temp = []
for key in domain_to_relations.keys():
    for rel in domain_to_relations[key]:
        temp.append((rel, key))
        
temp = sorted(temp, key=operator.itemgetter(0))

relation_to_domains = {}
for i in temp:
    if i[0] not in relation_to_domains:
        relation_to_domains[i[0]] = [i[1]]
    else:
        relation_to_domains[i[0]].append(i[1])
        
domains = list(domain_to_relations.keys())


"""----------------------ENRICHING-----------------------------------"""

query_enriching_probas = """select kbs_and.c1_synset_id as a,
                                count(distinct kbs_and.relation) as r,
                                count(kbs_and.c1_synset_id) as c
                                from kbs_and join domain 
                                on kbs_and.c1_synset_id=domain.synset_id
                                where domain_label= '%s' 
        							 group by a""" 
                                     
query_enriching_relation_from_synset = """select relation,
                                          count(relation) as c
                                          from kbs_and 
                                          where c1_synset_id='%s'
                                          group by relation
                                          order by c"""

query_enriching_question_struct = """select question, answer
                                     from struct 
                                     where upper(relation)='%s'
                                     and answer <> 'z'"""
                                     
query_enriching_tokens_retrieve_tokens = """select distinct c1_tokens 
                                            from kbs_and where c1_synset_id='%s'"""

query_enriching_retrieve_context = """select distinct context 
                                     from kbs_and where c1_synset_id = '%s'"""

query_enriching_not_in_db = """select b.tokens, b.synset_id 
                                from domain as a inner join senses as b 
                                on a.synset_id=b.synset_id 
                                where domain_label='%s'"""





"""---------------------QUERYING--------------------------------------"""
                                
query_querying_disambiguation_join = """select distinct tokens, 
                                                concept.synset_id, 
                                                domain_label
                                                from concept left join domain 
                                                on concept.synset_id=domain.synset_id 
                                                where tokens is not null 
                                                and role = 1 and punct(tokens) regexp \"^%s$\"""" 
                                            
query_sense_predominance =  """select c1, count(c1) 
                                from kbs where c1 in (%s) 
                                and question='%s' group by c1"""
                                
query_querying_question_struct = """select question, answer
                                     from struct 
                                     where relation='%s'
                                     and answer <> 'z'"""


"""---------------------------TURNS-----------------------------------"""

welcome = """Hey you, I'm willylorbot!\nI'm here to satisfy your thirst for knowledge.\nJust wake me up with a 'hey willy' and see what happens..."""
            
tell_me = """Yeah, tell me"""

willy_asks = """Wait wait wait... now it's my turn!"""

magic_words = """What are the magic words?"""

sleep = """Mmmm I want to sleep! YAAAAWN"""


"""--------------------NEW_KNOWLEDGE------------------------------"""

query_count_kbs = """select count(*) from kbs"""

query_check_new_knowledge = """ select distinct 
                            question, 
                            relation, 
                            answer, 
                            c1, 
                            c2, 
                            context, 
                            h_key,
                            substr(c1, 1 , instr(c1, ':')-1) as 'c1_tokens',
                            substr(c1, instr(c1, 'bn:')) as 'c1_synset_id',
                            substr(c2, 1 , instr(c2, ':')-1) as 'c2_tokens',
                            substr(c2, instr(c2, 'bn:')) as 'c2_synset_id'
                            from new_knowledge 
                            where question is not null 
                            and answer is not null
                            and relation is not null 
                            and context is not null
                            and question not like ''
                            and answer not like ''
                            and relation not like ''
                            and context not like ''
                            and (c1 like '%::bn:%' and c2 like '%::bn:%')"""
                            
query_insert_into_kbs = """insert into kbs select *, 
                                    (select max(incremental)+1 from kbs)
                                    from new_knowledge"""
                                
query_insert_into_kbs_and = """insert into kbs_and
                            select distinct 
                            question, 
                            relation, 
                            answer, 
                            c1, 
                            c2, 
                            context, 
                            h_key,
                            substr(c1, 1 , instr(c1, ':')-1) as 'c1_tokens',
                            substr(c1, instr(c1, 'bn:')) as 'c1_synset_id',
                            substr(c2, 1 , instr(c2, ':')-1) as 'c2_tokens',
                            substr(c2, instr(c2, 'bn:')) as 'c2_synset_id',
                            (select max(incremental)+1 from kbs_and)
                            from new_knowledge 
                            where question is not null 
                            and answer is not null
                            and relation is not null 
                            and context is not null
                            and question not like ''
                            and answer not like ''
                            and relation not like ''
                            and context not like ''
                            and (c1 like '%::bn:%' and c2 like '%::bn:%')"""
                            
def select_all_from_kbs_and(cursor):
    results = cursor.execute("select * from kbs_and")
    return results

