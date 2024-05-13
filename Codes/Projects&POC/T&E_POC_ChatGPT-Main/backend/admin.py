import os
import json
import pymongo
import logging
from datetime import datetime, timezone
import sys
# from dotenv import load_dotenv
# load_dotenv()

MONGO_CONNECTION_STRING = os.environ.get("MONGO_CONNECTION_STRING")

DEBUG = os.environ.get("DEBUG", "false")
if DEBUG.lower() == "true":
    logging.basicConfig(level=logging.DEBUG)

connection_string = 'mongodb+srv://analysis:Vector-dbstore@vectordbstore.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000'
mongodb_client = pymongo.MongoClient(connection_string)
db = mongodb_client["TravelandExpense"] 
Usercollection = db["UserDetails"] 
Entitycollection = db["Entities"]
Rolecollection =db["Roles"]
Querycollection =db["Queries"]

# Simple function to assist with vector search
def vector_search(query_embedding, num_results=5):
    
    embeddings_list = []
    pipeline = [
        {
            '$search': {
                "cosmosSearch": {
                    "vector": query_embedding,
                    "path": "QueryVector",
                    "k": num_results#, #, "efsearch": 40 # optional for HNSW only 
                    #"filter": {"title": {"$ne": "Azure Cosmos DB"}}
                },
                "returnStoredSource": True }},
        {'$project': { 'similarityScore': { '$meta': 'searchScore' }, 'document' : '$$ROOT' } }
    ]
    results = Querycollection.aggregate(pipeline)
    return results

def insertquery(request_body):
    try:  
        query ={}
        query = request_body                       
        queryexists = Querycollection.find_one(query)                        
        if queryexists:
            response = 'Query already exists'
        else:
            Querycollection.insert_one(request_body)
            response = 'Query inserted successfully'
        # return response
    except Exception as error:
        logging.exception("Exception while adding Query", error)

def insertentity(request_body):
    try:  
        query ={}
        query = request_body
        logging.info("Inserting entity",query)                        
        entityexists = Entitycollection.find_one(query)                        
        if entityexists:
            response = 'Entity already exists'
        else:
            Entitycollection.insert_one(request_body)
            response = 'Entity inserted successfully'
        return response
    except Exception as error:
        logging.exception("Exception while adding entity", error)

def insertrole(request_body):
    try: 
        query ={}
        query = request_body                               
        roleexists = Rolecollection.find_one(query)
        if roleexists:
            response = 'Role already exists'
        else:
            Rolecollection.insert_one(request_body)
            response = 'Role inserted successfully'
        return response
    except Exception as error:
        logging.exception("Exception while adding role", error)

def insertuser(request_body):
    try: 
        query ={}
        query = request_body                       
        userexists = Usercollection.find_one(query)
        if userexists:
            response = 'User already exists'
        else:                 
            Usercollection.insert_one(request_body)
            response = 'User inserted successfully'
        return response
    except Exception as error:
        logging.exception("Exception while adding user", error)

def getquerydetails(startdate,enddate): 
    try:         
        response =[]
        from_date = datetime.strptime(startdate, '%Y-%m-%d')
        to_date = datetime.strptime(enddate, '%Y-%m-%d')
        queries = Querycollection.find({"Datetime":{'$gte' : from_date,'$lte':to_date}},
                                       {"_id": 0,"QueryVector":0,"Content": 0,"Citation":0})
        for y in queries:
            response.append(y)        
        return response
    
    except Exception as error:
        logging.exception("Exception while getting query details", error)

def getentity(request_body):
    try:         
        response =[]
        if len(request_body)>0:
            
            query ={}
            query = request_body 
            logging.info("Req is not empty",query)                      
            mydoc = Entitycollection.find(query,{"_id": 0})
            for y in mydoc:
               response.append(y)
        else:
            logging.info("Req is empty")
            mydoc = Entitycollection.find({},{"_id": 0})
            for y in mydoc:
               response.append(y)
        
        return response
    except Exception as error:
        logging.exception("Exception while getting entity details", error)

def getrole(request_body):
    try:         
        response =[]
        if len(request_body)>0:
            query ={}
            query = request_body                       
            mydoc = Rolecollection.find(query,{ "_id": 0})
            for y in mydoc:
               response.append(y)
        else:
            mydoc = Rolecollection.find({},{ "_id": 0})
            for y in mydoc:
               response.append(y)
        
        return response
    except Exception as error:
        logging.exception("Exception while getting role details", error)

def getuser(request_body):
    try:         
        response =[]
        if len(request_body)>0:
            query ={}
            query = request_body                       
            mydoc = Usercollection.find(query,{ "_id": 0})
            for y in mydoc:
               response.append(y)
        else:
            mydoc = Usercollection.find({},{ "_id": 0})
            for y in mydoc:
               response.append(y)
        
        return response
    except Exception as error:
        logging.exception("Exception while getting user details", error)

def updatequery(request_body):
    try: 
        #selection_criteria = {}
        selection_criteria = request_body.get("SelectionCriteria")
        #updated_data = {}
        updated_data = request_body.get("UpdatedData")
        # queryexists = Querycollection.find_one(updated_data)                        
        # if queryexists:
        #     response = 'Query already exists'
        # else:
        Querycollection.update_one(selection_criteria,{"$set":updated_data}) 
        response = 'Query updated successfully'
        return response
        
    except Exception as error:
        logging.exception("Exception while updating query details", error)

def updateentity(request_body):
    try: 
        #selection_criteria = {}
        selection_criteria = request_body.get("SelectionCriteria")
        #updated_data = {}
        updated_data = request_body.get("UpdatedData")
        entityexists = Entitycollection.find_one(updated_data)                        
        if entityexists:
            response = 'Entity already exists'
        else:
            Entitycollection.update_one(selection_criteria,{"$set":updated_data}) 
            response = 'Entity updated successfully'
        return response
        
    except Exception as error:
        logging.exception("Exception while updating entity details", error)

def updateuser(request_body):
    try: 
        #selection_criteria = {}
        selection_criteria = request_body.get("SelectionCriteria")
        #updated_data = {}
        updated_data = request_body.get("UpdatedData")
        userexists = Usercollection.find_one(updated_data)
        if userexists:
            response = 'User already exists'
        else:  
            Usercollection.update_one(selection_criteria,{"$set":updated_data}) 
            response = 'User updated successfully'
        return response
        
    except Exception as error:
        logging.exception("Exception while updating user details", error)

def updaterole(request_body):
    try: 
        #selection_criteria = {}
        selection_criteria = request_body.get("SelectionCriteria")
        #updated_data = {}
        updated_data = request_body.get("UpdatedData")
        roleexists = Rolecollection.find_one(updated_data)
        if roleexists:
            response = 'Role already exists'
        else:
            Rolecollection.update_one(selection_criteria,{"$set":updated_data}) 
            response = 'Role updated successfully'
        return response
        
    except Exception as error:
        logging.exception("Exception while updating role details", error)