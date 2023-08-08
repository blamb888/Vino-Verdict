#first pip install streamlit
import streamlit as st


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
