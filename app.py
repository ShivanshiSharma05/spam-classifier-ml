import streamlit as st
from model import train_model, clean_text

# ================================
# 🔹 PAGE CONFIG
# ================================
st.set_page_config(page_title="Spam Classifier Pro", page_icon="📩")

# ================================
# 🔹 CUSTOM UI
# ================================
st.markdown("""
<style>
.main-box {
    background: linear-gradient(135deg, #1f2937, #111827);
    padding: 20px;
    border-radius: 15px;
}
.result {
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 20px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ================================
# 🔹 LOAD MODEL
# ================================
@st.cache_resource
def load():
    return train_model()

model, tfidf = load()

# ================================
# 🔹 HEADER
# ================================
st.markdown("""
<div class="main-box">
<h1>📩 Spam Message Classifier Pro</h1>
<p>Detect Spam Messages using Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# ================================
# 🔹 INPUT
# ================================
user_input = st.text_area("✍️ Enter your message:")

# ================================
# 🔹 PREDICT
# ================================
if st.button("🔍 Analyze Message"):

    if user_input.strip() == "":
        st.warning("⚠️ Enter a message")
    else:
        msg = clean_text(user_input)
        vector = tfidf.transform([msg])

        pred = model.predict(vector)[0]
        prob = model.predict_proba(vector)[0]
        confidence = max(prob) * 100

        if pred == 1:
            st.markdown(
                f'<div class="result" style="background:#7f1d1d;color:white;">🚨 SPAM<br>{confidence:.2f}% confidence</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result" style="background:#14532d;color:white;">✅ HAM<br>{confidence:.2f}% confidence</div>',
                unsafe_allow_html=True
            )

# ================================
# 🔹 FOOTER
# ================================
st.markdown("---")
st.caption("Built using Streamlit | ML Project")