import os
import json
import pymongo
import logging
import sys

DEBUG = os.environ.get("DEBUG", "false")
if DEBUG.lower() == "true":
    logging.basicConfig(level=logging.DEBUG)

connection_string = 'mongodb+srv://analysis:Password-1@vectordbstore.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000'
mongodb_client = pymongo.MongoClient(connection_string)
db = mongodb_client["ChatGPT"] 
Usercollection = db["UserDetails"] 
Entitycollection = db["Entities"]
Rolecollection =db["Roles"]

def insertentity(request_body):
    try:  
        query ={}
        query = request_body                       
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

def getentity(request_body):
    try:         
        response =[]
        if len(request_body)>0:
            query ={}
            query = request_body                       
            mydoc = Entitycollection.find(query,{"_id": 0})
            for y in mydoc:
               response.append(y)
        else:
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