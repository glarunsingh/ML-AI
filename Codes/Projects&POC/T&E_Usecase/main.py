import pymongo
from pymongo import MongoClient
import random
import re
from sentence_transformers import SentenceTransformer, util
import numpy as np
import json
import random


client = MongoClient("localhost", 27017)
db = client['test']


def insert_response():
    db.noresponse.insert_many([
    {"_id": 1, "response": "I’m sorry, but that’s like asking a toaster to make a salad. Let’s try something else!"},
    {"_id": 2, "response": "Oops! That question flew over my circuits. Can we try a ground ball instead?"},
    {"_id": 3, "response": "Hmm, that’s like asking a fish to juggle. Can we try another question?"},
    {"_id": 4, "response": "That’s a great question, but it’s like asking a cactus to sing. Can we try something else?"},
    {"_id": 5, "response": "I’m afraid that’s like asking a penguin to sunbathe. Can we navigate back to cooler waters?"},
    {"_id": 6, "response": "That’s outside my knowledge base. It’s like asking a snail to run a marathon!"},
    {"_id": 7, "response": "I’m sorry, but that’s like asking a robot to do a cartwheel. Can we try another question?"},
    {"_id": 8, "response": "That’s a bit like asking a cat to moo. They’re great at meowing, but mooing isn’t their thing. Can we try another question?"},
    {"_id": 9, "response": "Hmm, that’s like asking a squirrel to do your taxes. Can we explore another topic?"},
    {"_id": 10, "response": "I’m afraid that’s like asking a potato to dance. Can we try another question?"},
    {"_id": 11, "response": "That’s like asking a robot to cry at a sad movie. Can we try another question?"},
    {"_id": 12, "response": "Hmm, that’s like asking a turtle to do a backflip. Can we explore another topic?"},
    {"_id": 13, "response": "I’m sorry, but that’s like asking a chicken to bark. Let’s try something else!"},
    {"_id": 14, "response": "Oops! That question is like asking a tree to walk. Can we try a different one?"},
    {"_id": 15, "response": "That’s a great question, but it’s like asking a cloud to dance. Can we try something else?"},
    {"_id": 16, "response": "I’m afraid that’s like asking a rock to swim. Can we navigate back to solid ground?"},
    {"_id": 17, "response": "That’s a bit like asking a dog to meow. They’re great at barking, but meowing isn’t their thing. Can we try another question?"},
    {"_id": 18, "response": "Hmm, that’s like asking a rabbit to roar. Can we explore another topic?"},
    {"_id": 19, "response": "I’m sorry, but that’s like asking a robot to fall in love. Can we try another question?"},
    {"_id": 20, "response": "That’s a bit like asking a mouse to roar like a lion. Can we try another question?"}
    ])


#call the function to insert custom response in mangodb
# insert_response

json_response = {"id": "819871ff-2ddb-4b23-adad-fea481dd1ee5", "model": "gpt-35-turbo", "created": 1715776170, "object": "extensions.chat.completion", "choices": [{"messages": [{"role": "tool", "content": "{\"citations\": []}", "SPPath": "https://cognizantonline-my.sharepoint.com/:f:/r/personal/497299_cognizant_com/Documents/Desktop/WIP/T%26E%20-%20PD"}, {"role": "assistant", "content": "The requested information is not found in the retrieved data. Please try another query or topic."}]}], "history_metadata": {}}
#json_response = {"id": "ffcb835d-7fe4-47db-848f-96781264f94b", "model": "gpt-35-turbo", "created": 1715773860, "object": "extensions.chat.completion", "choices": [{"messages": [{"role": "tool", "content": "{\"citations\": [{\"content\": \"o Baggage wrapping charges. \\no Additional travel insurance not covered by the Corporate Card \\no Airline hospitality clubs v. Global Entry or TSA Pre-Check \\no In-flight headsets, entertainment, or travel kits \\no Any costs relating to usage of personal aircraft including, but not limited to fuel, insurance, payments \\n\\nto pilots. Usage of personal aircraft for business travel is explicitly prohibited. \\n\\n \\nBaggage \\n\\n \\nCognizant will cover a maximum of 4 bags / 100kg including any airline baggage allowance for the entire party \\ntraveling with the employee. The employee\\u2019s dependents are entitled to one bag (25kg) including any airline \\nbaggage allowance when they travel separately at a different time than the employee. \\n\\n \\nFor example, you are allowed to expense 4 pieces of baggage with a maximum combined weight of 100kg. If the \\nairline offers 1 free checked bag, employee can check 3 additional bags and expense for 3 pieces. If you are \\nchecking in a fifth bag, it is not reimbursable. If your bags are over the weight and/or size allowed by the airline, \\n\\n4 \\n\\nCognizant Confidential and/or Trade Secret \\nInternal Use Only \\n\\nDomestic Relocation Policy \\n\\n \\n\\n \\n\\n \\n \\n \\n\\n \\nany additional fees incurred are not reimbursable. \\n\\n \\n\\nAny additional excess baggage fee is at the employee\\u2019s own cost. Employees must weigh this cost against the \\npotential cost of shipping goods and choose the most cost-effective method. \\n\\nLease Breakage Policy \\n\\nPlease view the policy here. \\n\\nException Approval \\n\\nIn situations where an employee would like to reimburse additional reasonable relocation costs (shipping, flight, \\nor baggage), an exceptional approval will be required from the Account Lead/Client Partner and HR Lead in \\nAustralia. \\n\\n \\n \\n\\nQuestions \\n \\n\\nPlease reach out to Booking Travel (sharepoint.com) for further queries on Domestic Relocation. \\n\\nhttps://cognizantonline.sharepoint.com/sites/GlobalMobility/SitePages/International-Relocation.aspx#lease-breakage\\nhttps://cognizantonline.sharepoint.com/sites/CorporateFunctions/SitePages/n.aspx\\n\\n5 \\n\\nCognizant Confidential and/or Trade Secret \\nInternal Use Only \\n\\nDomestic Relocation Policy \\n\\n \\n\\n \\n\\n \\n \\n \\n \\n\\n2.0 India \\n \\n\\n2.1 Domestic Relocation \\n\\nIntroduction \\n\\nTo provide guidelines around the eligibility, terms, and conditions towards relocation of candidates and \\ntransfer of Cognizant associates. The policy is effective from 01 Sep 2019. \\n\\n \\nScope \\n\\n \\nThe policy covers: \\n\\n\\u2022 All India based associates transferred on permanent basis across Cognizant Technology \\nSolutions India Pvt. Ltd. and its Indian affiliate\\u2019s locations. \\n\\n\\u2022 Candidates joining Cognizant and are relocating from one location to another in India. \\n \\n\\nDefinitions \\n \\n\\nTAG: Talent Acquisition Group \\n \\n\\nDependent: Spouse or domestic partner co-habiting and minor children of Associate or spouse/partner \\nrelocating with the Associate \\n\\n \\nEntitlement Period: Period of 14 calendar days from the date of reaching the destination \\n\\nTransfer/relocation: Permanent relocation from one location to another Cognizant location within India \\n\\nTransfer/relocation: Permanent relocation from one location to another Cognizant location within India \\n\\nHRBP: Human Resources Business Partner \\n\\n \\nEligibility \\n\\n \\n\\u2022 All India based associates transferred on permanent basis across Cognizant-India locations. \\n\\u2022 Candidates joining Cognizant and are relocating from one location to another in India. \\n\\n \\nThe following table describes the entitlements under this policy: \\n\\n \\n \\n \\n\\nCategory Entitlement \\n\\n6 \\n\\nCognizant Confidential and/or Trade Secret \\nInternal Use Only \\n\\nDomestic Relocation Policy \\n\\n \\n\\n \\n\\n \\n \\n \\n\\n \\nTravel &amp; Accommodation Associate and three dependents will be entitled for \\n\\ntravel and stay as per Global Travel Policy. For details, \\nplease refer Navigation path\\nCognizant Confidential and/or Trade Secret\", \"id\": null, \"title\": \"Cognizant Confidential and/or Trade Secret\", \"filepath\": \"02_Domestic Relocation Policy.pdf\", \"url\": \"https://travelexpensedocspoc.blob.core.windows.net/fileupload-policyindex/02_Domestic%20Relocation%20Policy.pdf\", \"metadata\": null, \"chunk_id\": \"0\"}]}", "SPPath": "https://cognizantonline-my.sharepoint.com/:f:/r/personal/497299_cognizant_com/Documents/Desktop/WIP/T%26E%20-%20PD"}, {"role": "assistant", "content": "Based on the Domestic Relocation Policy of Cognizant, associates and candidates joining Cognizant and relocating are entitled to the reimbursement of expenses incurred towards transferring their personal effects and insurance costs, subject to specified limits. The reimbursement limits vary based on the level, with the highest level being eligible for INR 1,00,000 [doc1]. However, it is important to note that all expense claims, including movement of personal effects, should be submitted with relevant invoices, bills, and bank transaction proof [doc1]. Additionally, claims towards moving personal goods need to be raised within 60 days from the date of relocation [doc1].\n\nIf you have further questions about the Domestic Relocation Policy, it is recommended to reach out to the appropriate channels within Cognizant for further queries [doc1].\n\nPlease refer to the official policy documents for comprehensive information on the eligibility and process for goods transfer during relocation."}]}], "history_metadata": {}}
#json_response = {"id": "3669159a-6f87-4644-b969-973bab900fbd", "model": "gpt-35-turbo", "created": 1716287997, "object": "extensions.chat.completion", "choices": [{"messages": [{"role": "tool", "content": "{\\"citations\\": []}", "SPPath": "https://cognizantonline-my.sharepoint.com/:f:/r/personal/497299_cognizant_com/Documents/Desktop/WIP/T%26E%20-%20PD"}, {"role": "assistant", "content": "This question is out of the scope of the retrieved documents."}]}], "history_metadata": {}}
# Function to extract content where role is assistant
def extract_assistant_content(json_response):
    messages = json_response.get("choices", [])[0].get("messages", [])
    assistant_contents = [message["content"] for message in messages if message["role"] == "assistant"]
    return assistant_contents


# Extract content for both JSON responses
content = extract_assistant_content(json_response)

# Print the extracted content
print("Content from JSON Response:")
print(content)

#Defining the keywords in the response from LLM for no data
# Predefined "no data" keywords
no_data_keywords = [
    "not found",
    "no results",
    "empty",
    "no matches",
    "couldn't find any relevant data",
    "0",
    "This question is out of domain.",
    "Based on the information provided, the eligibility for goods transfer during relocation is not explicitly mentioned in the retrieved documents. Therefore, I cannot provide a specific answer regarding the eligibility for goods transfer during relocation based on the retrieved documents. If you have specific questions about the relocation policy, it would be best to refer to the official policy documents or reach out to the relevant department for clarification.",
    "Based on the information provided, you can claim the expenses incurred during your business trip of 14 days, which exceeded 2 days for business requirement, under the 'Repatriation Claims - Process (Global)' policy. This policy covers expenses related to repatriation, including baggage, hotel, COVID-19 medical tests, ground transportation, individual meals, and other related expenses. It provides detailed guidelines and supporting documents required for claiming various types of expenses during such trips.",
    "The requested information is not found in the retrieved data. Please try another query or topic.",
    "The retrieved documents do not contain specific information on how to change the policy in Concur. Therefore, I am unable to provide a direct answer based on the retrieved documents.",
    "Error Message",
    "This question is out of the scope of the retrieved documents.",
    "The requested information is not found in the retrieved data. Please try another query or topic.-No guidelines for Depedents relocation Airfare expenses found",
    "The SOX documentation to be followed is not specified in the retrieved documents. Therefore, the specific SOX documentation to be followed is not available in the provided documents.",
    "Based on the retrieved documents, there is no specific mention of the frequency with which reports should be run to ascertain the amount outstanding with Amex. Therefore, the exact number of times in a week to run the reports for this purpose is not provided in the documents.",
    "This is out of Domain."
]

#1. Keyword pattern to define no data response
def no_keyword_detection(responses, keywords):
    #keyword_patterns = [re.compile(r'\b{}\b'.format(keyword), re.IGNORECASE) for keyword in keywords]
    keyword_patterns = [re.compile(re.escape(keyword), re.IGNORECASE) for keyword in keywords]
    
    
    for response in responses:
        for pattern in keyword_patterns:
            if pattern.search(response):
                return False  # No data found
    return True  # Data found

#2. ML - Initialize the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode the predefined keywords
keyword_embeddings = model.encode(no_data_keywords)

def ml_sentencetransformation_model(response, threshold=0.75):
    # Encode the response
    response_embedding = model.encode(response)
    
    # Compute cosine similarities
    cosine_scores = util.cos_sim(response_embedding, keyword_embeddings)[0]
    
    # Check if any similarity score is above the threshold
    return any(score > threshold for score in cosine_scores)


def is_citation_empty(data):
    
    # Extract the content of the "tool" message
    tool_message_content = None
    for choice in data.get("choices", []):
        for message in choice.get("messages", []):
            if message.get("role") == "tool":
                tool_message_content = message.get("content")
                break
    
    # If the tool message content is found, parse it
    if tool_message_content:
        tool_content_data = json.loads(tool_message_content)
        # Check if the citations list is empty
        return len(tool_content_data.get("citations", [])) == 0
    
    # Return False if the citations field is not found
    return False


# Get random response from Mongodb
def get_random_response(db):
    # Fetch all responses from MongoDB
    responses = list(db.noresponse.find({}, {'_id': 0, 'response': 1}))
    # Select a random response
    random_response = random.choice(responses)
    return random_response['response']
    
    #return response_from_llm  # If response from LLM is not empty, return it


#main function

#Validating Keyword detection and ML model and Citation
if no_keyword_detection(content, no_data_keywords) == True and ml_sentencetransformation_model(content) == False and is_citation_empty(json_response) == False:
    print(no_keyword_detection(content, no_data_keywords))
    print(ml_sentencetransformation_model(content))
    print(is_citation_empty(json_response))
    response_from_llm = content
    print(response_from_llm)
    
else:
    print(no_keyword_detection(content, no_data_keywords))
    print(ml_sentencetransformation_model(content))
    print(is_citation_empty(json_response))
    response = get_random_response(db)
    print(response)

    for choice in json_response["choices"]:
        for message in choice["messages"]:
            if message["role"] == "assistant":
                 message["content"] = response

json_response
