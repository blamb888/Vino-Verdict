#first pip install streamlit
import streamlit as st

from transformers import BertTokenizer, BertForSequenceClassification
import torch


#---------------------------------- first draft--------
st.title('Vino Verdict test page')


st.subheader('Recommendation level based on wine reviews -- ??? ')

#st.caption ('any caption here if needed')

#st.latex(" x^2 math things here")

#model = loadMySuperModel()


st.subheader('INPUT BOX: input some wine label/review #(to be decided- which one)')



review=st.text_input("Review or label")



st.button("Check if recommended")
st.write(review) #for now



#if st.button("Check if recommended"):
#    label = model.predict(review)
#    st.write(label)


#-------------------------------------------------------

# Load model and tokenizer
# bin file should be named as pytorch_model.bin, and config.json should be in the folder
MODEL_PATH = "/Users/sultanl/code/blamb888/Vino-Verdict"

model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
#model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

######## add preprocessing functions here

st.title('Wine Review Predictor')

# Get user input
review = st.text_area("Enter wine review:")

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

    # Display prediction
    st.write(f'Predicted Rating Category: {prediction}')
