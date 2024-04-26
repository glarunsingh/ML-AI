from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from secret_key import openapi_key

import os
os.environ['OPENAI_API_KEY']= openapi_key

llm = OpenAI(temperature=0.7)

def generate_restaurant_name_and_items(cuisine):
    #Chain 1: Restaurant Name
    prompt_template_name = PromptTemplate(
    input_variables=['cuisine'],
    template='I want to open a restaurant for {cuisine} food. Suggest a name.'                     
    )

    name_chain = LLMChain(prompt=prompt_template_name, llm=llm, output_key='res_name')

    #Chain 2: Menu Items
    prompt_template_items = PromptTemplate(
    input_variables=['res_name'],
    template='Suggest some menu items for {res_name}. Return it as a comma separated list.'                     
    )

    food_items_chain = LLMChain(prompt=prompt_template_items, llm=llm, output_key='menu_items')

    chain = SequentialChain(
        chains=[name_chain, food_items_chain],
        input_variables = ['cuisine'],
        output_variables = ['res_name', 'menu_items']    
    )

    response = chain({'cuisine': cuisine})

    return response

if __name__ == "__main__":
    print(generate_restaurant_name_and_items("Indian"))