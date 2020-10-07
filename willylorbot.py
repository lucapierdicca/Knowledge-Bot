import json 
import requests


last_update_id = None

TOKEN = "436364046:AAGwZfEVom7X1I1gf_JwMtljmFmSIwXJWFw"
URL = "https://api.telegram.org/bot{}/".format(TOKEN)


def get_url(url):
    response = requests.get(url)
    content = response.content.decode("utf8")
    return content


def get_json_from_url(url):
    content = get_url(url)
    js = json.loads(content)
    return js


def get_updates(offset=None):
    url = URL + "getUpdates?timeout=100"
    if offset:
        url += "&offset={}".format(offset)
    js = get_json_from_url(url)
    return js

def get_last_update_id(updates):
    update_ids = []
    for update in updates["result"]:
        update_ids.append(int(update["update_id"]))
    return max(update_ids)


def get_last_chat_id_and_text(updates):
    num_updates = len(updates["result"])
    last_update = num_updates - 1
    text = updates["result"][last_update]["message"]["text"]
    chat_id = updates["result"][last_update]["message"]["chat"]["id"]
    return (text, chat_id)



def send_message(text, chat_id):
    url = URL + "sendMessage?text={}&chat_id={}".format(text, chat_id)
    get_url(url)
    

def generate_answer(updates, restored_r, restored_c1c2, restored_Atype):
    
    for update in updates["result"]:
        question = update["message"]["text"]
        chat = update["message"]["chat"]["id"]
        answer = ''

    return answer, chat
    

def willy_answers(answer, chat):
        try:
            send_message(answer, chat)
        except Exception as e:
            print(e)
    



