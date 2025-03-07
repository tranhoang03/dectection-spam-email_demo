import streamlit as st
import pandas as pd
import joblib
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import standard_data

# Đọc dữ liệu và tải mô hình
@st.cache_resource
def load_model_and_vectorizer():
    model_en = joblib.load('https://raw.githubusercontent.com/tranhoang03/dectection-spam-email_demo/master/svm_model_en.joblib')
    vectorizer_en = joblib.load('https://raw.githubusercontent.com/tranhoang03/dectection-spam-email_demo/master/vectorizer_en.joblib')
    model_vi = joblib.load('https://raw.githubusercontent.com/tranhoang03/dectection-spam-email_demo/master/svm_model_vi.joblib')
    vectorizer_vi = joblib.load('https://raw.githubusercontent.com/tranhoang03/dectection-spam-email_demo/master/vectorizer_vi.joblib')

    return model_en, vectorizer_en, model_vi, vectorizer_vi

model_en, vectorizer_en, model_vi, vectorizer_vi = load_model_and_vectorizer()

# Hàm xử lý văn bản
def process_text(text, language):
    try:
        if language == "en":
            text = standard_data.standard_en(text)
            input_data = vectorizer_en.transform([text])
            prediction = model_en.predict_proba(input_data)
        else:
            text = standard_data.standard_vi(text)
            input_data = vectorizer_vi.transform([text])
            prediction = model_vi.predict_proba(input_data)

        label = "Spam mail" if prediction[0][1] > 0.5 else "Ham mail"
        probability = prediction[0][1] if label == "Spam mail" else prediction[0][0]
        return f"{label} ({probability:.2f})"
    except Exception as e:
        return str(e)

# Gửi email
def send_email(sender_email, sender_password, recipient_email, email_content):
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)

            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = recipient_email
            msg['Subject'] = "Forwarded Email Content"
            msg.attach(MIMEText(email_content, 'plain'))

            server.sendmail(sender_email, recipient_email, msg.as_string())
        return "Email sent successfully!"
    except Exception as e:
        return str(e)

# CSS để tùy chỉnh giao diện và hiển thị tên tác giả
st.markdown("""
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f4f7f6;
            color: #333;
        }
        .main-title {
            font-size: 36px;
            color: #2c3e50;
            text-align: center;
            font-weight: bold;
        }
        .description {
            font-size: 18px;
            color: #7f8c8d;
            text-align: center;
            margin-bottom: 30px;
        }
        .stTextArea, .stTextInput {
            border: 2px solid #bdc3c7;
            border-radius: 10px;
            margin-top: 10px;
        }
        .stButton>button {
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #2980b9;
            cursor: pointer;
        }
        .result-box {
            border: 1px solid #2ecc71;
            border-radius: 10px;
            padding: 15px;
            background-color: #eafaf1;
            color: #27ae60;
            text-align: center;
            margin-top: 20px;
            font-size: 18px;
        }
        .error-box {
            border: 1px solid #e74c3c;
            border-radius: 10px;
            padding: 15px;
            background-color: #fdecea;
            color: #c0392b;
            text-align: center;
            margin-top: 20px;
            font-size: 18px;
        }
        .author-info {
            position: fixed;
            bottom: 10px;
            right: 10px;
            font-size: 14px;
            color: #95a5a6;
            text-align: right;
        }
    </style>
""", unsafe_allow_html=True)

# Giao diện chính
st.markdown('<div class="main-title">📧 Email Spam Checker</div>', unsafe_allow_html=True)


# Giao diện nhập liệu
with st.form(key="email_form"):
    email_content = st.text_area("✏️ Enter email content:", "", height=200)
    language = st.selectbox("🌐 Choose language:", ["English", "Vietnamese"], index=0)
    submit_button = st.form_submit_button("🚀 Check Spam")

if submit_button:
    result = process_text(email_content, "en" if language == "English" else "vi")
    if "Ham mail" in result:
        st.markdown(f'<div class="result-box">✅ {result}</div>', unsafe_allow_html=True)
        st.session_state["is_spam_checked"] = True
        st.session_state["email_content"] = email_content
    else:
        st.markdown(f'<div class="error-box">🚫 {result}</div>', unsafe_allow_html=True)
        st.session_state["is_spam_checked"] = False

# Gửi email sau khi kiểm tra spam thành công
if st.session_state.get("is_spam_checked", False):
    st.markdown("### ✉️ Send Your Email")
    sender_email = st.text_input("📧 Your email:")
    sender_password = st.text_input("🔑 Your password:", type="password")
    recipient_email = st.text_input("📨 Recipient's email:")

    if st.button("📤 Send Email"):
        email_content = st.session_state.get("email_content", "")
        send_result = send_email(sender_email, sender_password, recipient_email, email_content)
        if "successfully" in send_result:
            st.markdown('<div class="result-box">✅ Email sent successfully!</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="error-box">🚫 {send_result}</div>', unsafe_allow_html=True)

# Hiển thị tên tác giả và mã sinh viên
st.markdown("""
    <div class="author-info">
        Developed by: <strong>Trần Văn Hoàng</strong> <br>
        Student ID: <strong>21T1020871</strong>
    </div>
""", unsafe_allow_html=True)
