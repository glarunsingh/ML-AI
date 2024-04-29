import os
import json
import logging
import requests
import dataclasses
import re


DEBUG = os.environ.get("DEBUG", "false")
if DEBUG.lower() == "true":
    logging.basicConfig(level=logging.DEBUG)

AZURE_SEARCH_PERMITTED_GROUPS_COLUMN = os.environ.get("AZURE_SEARCH_PERMITTED_GROUPS_COLUMN")

class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)

async def format_as_ndjson(r):
    try:
        async for event in r:
            yield json.dumps(event, cls=JSONEncoder) + "\n"
    except Exception as error:
        logging.exception("Exception while generating response stream: %s", error)
        yield json.dumps({"error": str(error)})

def parse_multi_columns(columns: str) -> list:
    if "|" in columns:
        return columns.split("|")
    else:
        return columns.split(",")


def fetchUserGroups(userToken, nextLink=None):
    # Recursively fetch group membership
    if nextLink:
        endpoint = nextLink
    else:
        endpoint = "https://graph.microsoft.com/v1.0/me/transitiveMemberOf?$select=id"
    
    headers = {
        'Authorization': "bearer " + userToken
    }
    try :
        r = requests.get(endpoint, headers=headers)
        if r.status_code != 200:
            logging.error(f"Error fetching user groups: {r.status_code} {r.text}")
            return []
        
        r = r.json()
        if "@odata.nextLink" in r:
            nextLinkData = fetchUserGroups(userToken, r["@odata.nextLink"])
            r['value'].extend(nextLinkData)
        
        return r['value']
    except Exception as e:
        logging.error(f"Exception in fetchUserGroups: {e}")
        return []


def generateFilterString(userToken):
    # Get list of groups user is a member of
    userGroups = fetchUserGroups(userToken)

    # Construct filter string
    if not userGroups:
        logging.debug("No user groups found")

    group_ids = ", ".join([obj['id'] for obj in userGroups])
    return f"{AZURE_SEARCH_PERMITTED_GROUPS_COLUMN}/any(g:search.in(g, '{group_ids}'))"

def format_non_streaming_response(chatCompletion, history_metadata, message_uuid=None):
    response_obj = {
        "id": chatCompletion.id,
        "model": chatCompletion.model,
        "created": chatCompletion.created,
        "object": chatCompletion.object,
        "choices": [
            {
                "messages": []
            }
        ],
        "history_metadata": history_metadata
    }

    if len(chatCompletion.choices) > 0:
        message = chatCompletion.choices[0].message
        if message:
            if hasattr(message, "context") and message.context.get("messages"):
                for m in message.context["messages"]:
                    if m["role"] == "tool":                     
                        response_obj["choices"][0]["messages"].append({
                            "role": "tool",
                            "content": m["content"],
                            "SPPath":"https://cognizantonline-my.sharepoint.com/:f:/r/personal/497299_cognizant_com/Documents/Desktop/WIP/T%26E%20-%20PD"
                        })
            elif hasattr(message, "context"):
                response_obj["choices"][0]["messages"].append({
                    "role": "tool",
                    "content": json.dumps(message.context),
                    "SPPath":"https://cognizantonline-my.sharepoint.com/:f:/r/personal/497299_cognizant_com/Documents/Desktop/WIP/T%26E%20-%20PD",
                })
            response_obj["choices"][0]["messages"].append({
                "role": "assistant",
                "content": message.content,
            })

            referencecontent = json.loads(response_obj["choices"][0]["messages"][0]["content"])["citations"]
            responsecontent = response_obj["choices"][0]["messages"][1]["content"]
            # print(referencecontent)
            # print(responsecontent)
            referencecontent,responsecontent = format_response(referencecontent,responsecontent)
            response_obj["choices"][0]["messages"][1]["content"] = responsecontent
            x = {}
            x['citations']=referencecontent
            response_obj["choices"][0]["messages"][0]["content"] = json.dumps(x)
            return response_obj
    
    return {}

def format_response(referencecontent,responsecontent):
    #Getmapping between file and doc
    mappinglst = getmapping(referencecontent,'filepath')

    #Get duplicate filelist and mark them for deletion
    deletelist = find_duplicates(referencecontent, 'filepath')
    # print(deletelist)

    #Replace duplicate docid's with originalid in Content
    responsecontent = replace_duplicates_content(deletelist,responsecontent)
    #print(responsecontent)

    #Remove duplicate docid's in mapping list
    remove_duplicates_mapping(mappinglst,deletelist)
    #print(mappinglst)

    #Fetch unique doc list from content
    doclist_content = re.findall("doc\d{1}", responsecontent)
    doclist_content_unique = fetch_unique_items(doclist_content)
    #print(doclist_content_unique)

    #get updated mapping list     
    mapping_updated = get_mapping_updated(mappinglst,doclist_content_unique,deletelist)
    # print(mapping_updated)
    # print(deletelist)

    #Delete the  unreferenced file list from reference content
    referencecontent = update_reference(deletelist,referencecontent)
    #print(referencecontent)

    #Update the content as per pendingn list of files
    responsecontent = update_content(mapping_updated,responsecontent)
    #print(responsecontent)
    return referencecontent,responsecontent   

def getmapping(data,key):
    mapping =[]   
    docid=1
    for index, dictionary in enumerate(data):     
        value = dictionary.get(key)
        mapping.append({
        'filename': value,
        'fileid': index+1,
        'docid': f'doc{docid}'            
        })
        
        docid+=1
    #print(mapping)
    return mapping


def find_duplicates(lst, key):
    seen = []
    duplicates = []
    
    for index, dictionary in enumerate(lst):
        value = dictionary.get(key)
        
        exists = check_value_in_list_of_dicts(seen,'filename',value)
        
        if exists:
                duplicates.append({
                    'filename': value,
                    'removefileid': index+1,
                    'originalfileid':fetch_value_in_list_of_dicts(seen, 'filename', value,'fileid')
                })
        else:
                seen.append({
                    'filename': value,
                    'fileid': index+1                            
            })
    
    return duplicates

def check_value_in_list_of_dicts(lst, key, value):
    for d in lst:
        if d.get(key) == value:
            return True
    return False

def fetch_value_in_list_of_dicts(lst, searchkey, value,fetchkey):
    for d in lst:
        if d.get(searchkey) == value:
            return d.get(fetchkey)
        
def replace_duplicates_content(data,txt):    
    for d in data:
        removedocid =d.get('removefileid')
        originaldocid = d.get('originalfileid')
        txt = txt.replace(f'doc{removedocid}',f'doc{originaldocid}')
    return txt
    

# def update_value_in_list_of_dicts(lst, key, value, new_value):
#         for d in lst:
#             if d.get(key) == f'doc{value}':
#                 d[key] = f'doc{new_value}'

def remove_duplicates_mapping(mappinglst,deletelist):     
    for dictionary in deletelist:
        value = dictionary.get('removefileid')
        newvalue =dictionary.get('originalfileid')
        delete_dict_item(mappinglst, 'fileid', value)
        #update_value_in_list_of_dicts(mappinglst,'docid',value,newvalue)

def delete_dict_item(lst, key, value):
        for d in lst:
            if d.get(key) == value:
                lst.remove(d)
                break

def fetch_unique_items(doclist):
    uniquelist =[]
    for a in doclist:
#         # check if exists in unique_list or not
            if a not in uniquelist:
                uniquelist.append(a)
    return uniquelist   
           
def get_mapping_updated(mappinglst,doclist,deletelist):
    mapping_updated =[]   
    for index, dictionary in enumerate(mappinglst):                
            value = dictionary.get('filename')
            fileid = dictionary.get('fileid')
            docid = dictionary.get('docid')
            if docid in doclist:
               mapping_updated.append({
                    'filename': value,
                    'fileid': fileid,
                    'docid': docid         
                    }) 
            else:
                deletelist.append({                     
                    'filename': value,
                    'removefileid': fileid,
                })             
                
                       
    return mapping_updated

def update_reference(dellist,ref):
    ref_updated = []
    for index, dictionary in enumerate(ref): 
        value = index +1
        exists = check_value_in_list_of_dicts(dellist,'removefileid',value) 
        if not exists:
           ref_updated.append(dictionary)
    return ref_updated 
         
        #   delete_dict_item_by_index(ref,index)

# def delete_dict_item_by_index(lst, index):
#         if index >= 0 and index < len(lst):
#             del lst[index]     

def update_content(data,txt):
     for index, dictionary in enumerate(data):     
         value = dictionary.get('docid')
         updatedvalue =index +1
         #print(updatedvalue)
         txt = txt.replace(value,f'doc{updatedvalue}')
     return txt

def format_stream_response(chatCompletionChunk, history_metadata, message_uuid=None):
    response_obj = {
        "id": chatCompletionChunk.id,
        "model": chatCompletionChunk.model,
        "created": chatCompletionChunk.created,
        "object": chatCompletionChunk.object,
        "choices": [{
            "messages": []
        }],
        "history_metadata": history_metadata
    }

    if len(chatCompletionChunk.choices) > 0:
        delta = chatCompletionChunk.choices[0].delta
        if delta:
            if hasattr(delta, "context") and delta.context.get("messages"):
                for m in delta.context["messages"]:
                    if m["role"] == "tool":
                        messageObj = {
                            "role": "tool",
                            "content": m["content"]
                        }
                        response_obj["choices"][0]["messages"].append(messageObj)
                        return response_obj
            if delta.role == "assistant" and hasattr(delta, "context"):
                messageObj = {
                    "role": "assistant",
                    "context": delta.context,
                }
                response_obj["choices"][0]["messages"].append(messageObj)
                return response_obj
            else:
                if delta.content:
                    messageObj = {
                        "role": "assistant",
                        "content": delta.content,
                    }
                    response_obj["choices"][0]["messages"].append(messageObj)
                    return response_obj
    
    return {}