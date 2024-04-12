import streamlit as st
import langchain_helper

st.title("Restaurant Name Generator")

cuisine = st.sidebar.selectbox("Pick a Cusines", ("Indian", "American", "Chinese", "Japanese", "Arabic", "Mexican"))



if cuisine:
    response = langchain_helper.generate_restaurant_name_and_items(cuisine)
    st.header(response['res_name'].strip())
    menu_items = response['menu_items'].strip().split(",")
    st.write("**Menu Items**")
    for items in menu_items:
        st.write(items)