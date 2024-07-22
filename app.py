import random
import streamlit as st
import random
import json
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

intents=json.load(open("intents.json"))
patterns=[]
tags=[]
for i in intents["intents"]:
    for j in i["patterns"]:
        patterns.append(j)
        tags.append(i["tag"])
print(len(tags))
print(len(patterns))

tf=TfidfVectorizer()
sacaled_pattern=tf.fit_transform(patterns)
model=LogisticRegression() 
model.fit(sacaled_pattern,tags)


# input_message=input("enter message")
# input_message=tf.transform([input_message])

def chatbox(input_message):
    input_message=tf.transform([input_message])
    pred_tag=model.predict(input_message)[0]
    for i in intents["intents"]:
        if i["tag"]==pred_tag:
            responses=random.choice(i["responses"])
            return responses


st.title("UNIVERSITY BOAT")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = f"AI Boat: "+ chatbox(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
