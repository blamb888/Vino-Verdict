#first pip install streamlit
from google.cloud import storage

# Initialise a client
storage_client = storage.Client("graphic-armor-392809")
# Create a bucket object for our bucket
bucket = storage_client.bucket('vino-verdict')
# Create a blob object from the filepath
blob = bucket.blob("models/pytorch_model.bin")
# Download the file to a destination
blob.download_to_filename("model/pytorch_model.bin")

blob2 = bucket.blob("models/config.json")
blob2.download_to_filename("model/config.json")


import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from PIL import Image



# Load model and tokenizer
##pretrained model from bert
##model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# bin file should be named as pytorch_model.bin, and config.json should be in the project folder

MODEL_PATH = "model"
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

######## add preprocessing functions here

st.title('Wine Review Predictor')
vino=Image.open('./members/vino.jpeg')
st.image(vino)

# Get user input
review = st.text_area("Enter a wine review:", height=10)

if st.button("Predict"):
    # Tokenize and preprocess input
    inputs = tokenizer.encode_plus(
        review,
        add_special_tokens=True,
        max_length=128,
        return_tensors='pt'
    )
    with torch.no_grad():
        # Predict
        logits = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])[0]
        prediction = torch.argmax(logits, dim=1).item()

    # Display prediction  #Predicted Rating Category
    if prediction==0:
        st.write('Sentiment Result: Bad Wine!')
    if prediction==1:
        st.write('Sentiment Result: Good Wine!')
    if prediction==2:
        st.write('Sentiment Result: Exellent Wine! :sunglasses:')





#project members
st.divider()
st.subheader("Vino-Verdict members")
col1, col2, col3 = st.columns(3, gap='medium')


bl=Image.open('./members/bl.jpg')
sl=Image.open('./members/sl.jpg')
ao=Image.open('./members/ao.jpg')
with col1:
    st.image(bl,caption='Team leader: Brandon')
with col2:
    st.image(sl,caption='Team member: Sultan')
with col3:
    st.image(ao,caption='Team member: Ayata')
