import requests
import json
import const

url = "http://151.100.179.26:8080/KnowledgeBaseServer/rest-api/"
key = "key=58abb1ed-8886-4957-9323-330facfceee3"

#line=[]
#f = open('kbs.tsv', 'a')
#for i in range(0, 1129044, 5000):
#	data = requests.get('http://151.100.179.26:8080/KnowledgeBaseServer/rest-api/items_from?id='+str(i)+'&key=58abb1ed-8886-4957-9323-330facfceee3')
#	data = json.loads(data.text)
#	for j in data:
#		for key, value in j.items():
#			line.append(str(j[key]))
#		f.write('|'.join(line)+'\n')
#		del line[:]
#
#f.close()


def retrieve_new_entries(cursor, conn):
    current_kbs_rows = cursor.execute(const.query_count_kbs).fetchall()[0][0]
    current_online_kbs_rows = json.loads(requests.get(url+'items_number_from?id=0&'+key).text)
    
    delta = int(current_online_kbs_rows) - int(current_kbs_rows)
    
    if delta > 50:
        data = json.loads(requests.get(url+"items_from?id=%d&" % (current_kbs_rows) +key ).text)
        record = [(entry['question'],entry['answer'],entry['relation'],entry['context'],str(entry['domains']),entry['c1'],entry['c2'],entry['HASH']) for entry in data] 
        cursor.executemany("insert into new_knowledge values (?,?,?,?,?,?,?,?)", record)
        results = cursor.execute(const.query_check_new_knowledge).fetchall()
        
        if len(results) >= 50:
            cursor.execute(const.query_insert_into_kbs_and)
            cursor.execute(const.query_insert_into_kbs)
            cursor.execute("delete from new_knowledge")
            conn.commit()
        
            return 1
        else:
            cursor.execute("delete from new_knowledge")
            conn.commit()
            
            return 0
        

def send_entry(entry):
    json_data = json.dumps(entry)
    r_value = requests.post(url+"add_item_test?"+key, data=json_data)
    
    return r_value
        
