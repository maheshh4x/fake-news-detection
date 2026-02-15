import streamlit as st

st.markdown(
    """
    <style>
    .centered-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 80vh;  /* Adjust the height as needed */
        flex-direction: column;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.text('')
st.text('')
st.text('')
st.text('')
st.text('')
st.text('')
st.text('')
st.title('👋 Welcome to Fake News Detection Tools!')
st.markdown("""
    <style>
    .justified-text {
        text-align: justify;
    }
    </style>
    <div class="justified-text">
        Identifying fake news has become increasingly difficult as it has turned into an industry where individuals are paid to write sensational stories and create clickbait content to drive traffic. 
         Misleading and intentionally false information is often made to appear credible, reaching thousands of users within minutes. To combat this, our Fake News Detection Tools provide you with the necessary resources to quickly identify and verify the authenticity of news content.
    </div>
    <br>
    """, unsafe_allow_html=True)

with st.container():
    st.text('')
    
    st.write('''For more technical details, click on **Tools Information** or proceed directly to the tools by selecting **Start**''')
    col1, col2, col3, col4 = st.columns(4, gap="small")
    if col1.button("Tools Information"):
        st.switch_page("fake_news_detection/information.py")
    if col2.button("Start"):
        st.switch_page("fake_news_detection/tools.py")
        