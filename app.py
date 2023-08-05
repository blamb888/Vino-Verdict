#first pip install streamlit
import streamlit as st

st.write('Trying if I can make streamlit work')

st.title('this will be title')

st.subheader('subheader here!!')

st.caption ('any caption here if needed')

st.latex(" x^2 math things here")

#model = loadMySuperModel()

st.subheader('INPUT BOX: input some wine label/review #(decide which one)')

review=st.text_input("Review or label")

st.button("Check if recommended")
st.write(review) #for now



#if st.button("Check if recommended"):
#    label = model.predict(review)
#    st.write(label)
