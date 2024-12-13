import streamlit as st
import pandas as pd
import joblib
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import imaplib
import email
from email.header import decode_header
import standard_data

# Đọc dữ liệu và tải mô hình
@st.cache_resource
def load_model_and_vectorizer():
    data_en = pd.read_csv('data_en.csv')
    data_vi = pd.read_csv('data_vi.csv', encoding="utf-8")

    model_en = joblib.load('svm_model_en.joblib')
    vectorizer_en = joblib.load('vectorizer_en.joblib')
    model_vi = joblib.load('svm_model_vi.joblib')
    vectorizer_vi = joblib.load('vectorizer_vi.joblib')

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

# Lấy email từ Gmail
def fetch_email(email_account, email_password):
    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(email_account, email_password)
        mail.select("inbox")

        status, messages = mail.search(None, "ALL")
        email_ids = messages[0].split()
        latest_email_id = email_ids[-1]

        status, msg_data = mail.fetch(latest_email_id, "(RFC822)")
        for response_part in msg_data:
            if isinstance(response_part, tuple):
                msg = email.message_from_bytes(response_part[1])
                email_subject, encoding = decode_header(msg["Subject"])[0]
                if isinstance(email_subject, bytes):
                    email_subject = email_subject.decode(encoding or "utf-8")

                if msg.is_multipart():
                    for part in msg.walk():
                        content_type = part.get_content_type()
                        if content_type == "text/plain":
                            email_body = part.get_payload(decode=True).decode()
                            break
                else:
                    email_body = msg.get_payload(decode=True).decode()

        mail.close()
        mail.logout()
        return email_subject, email_body
    except Exception as e:
        return None, str(e)

# CSS để tùy chỉnh giao diện
st.markdown("""
    <style>
        .main-title {
            font-size: 30px;
            color: #2c3e50;
            text-align: center;
        }
        .description {
            font-size: 16px;
            color: #7f8c8d;
            text-align: center;
            margin-bottom: 20px;
        }
        .stButton>button {
            background-color: #3498db;
            color: white;
            border-radius: 8px;
        }
        .stTextInput, .stTextArea {
            border: 1px solid #bdc3c7;
        }
    </style>
""", unsafe_allow_html=True)

# Xây dựng giao diện với Streamlit
st.markdown("""
    <div class="main-title">📧 Email Spam Checker</div>
    <div class="description">
        Use this app to detect whether an email is spam or ham
    </div>
""", unsafe_allow_html=True)

# Lưu trạng thái trong session_state
if "email_body" not in st.session_state:
    st.session_state.email_body = ""
if "email_subject" not in st.session_state:
    st.session_state.email_subject = ""

option = st.selectbox(
    "What do you want to do?",
    ["✏️ Write Email", "📥 Get Email from My Gmail"]
)

if option == "✏️ Write Email":
    col1, col2 = st.columns(2)
    with col1:
        email_content = st.text_area("Enter email content:", "")
    with col2:
        language = st.selectbox("Choose language:", ["English", "Vietnamese"])

    if st.button("Check Spam"):
        result = process_text(email_content, "en" if language == "English" else "vi")
        if "Ham mail" in result:
            st.success(f"✅ {result}")
            st.session_state["is_spam_checked"] = True
            st.session_state["email_content"] = email_content
        else:
            st.error(f"🚫 {result}")
            st.session_state["is_spam_checked"] = False

    if st.session_state.get("is_spam_checked", False):
        sender_email = st.text_input("Your email:")
        sender_password = st.text_input("Your password:", type="password")
        recipient_email = st.text_input("Recipient's email:")

        if st.button("Send Email"):
            email_content = st.session_state.get("email_content", "")
            send_result = send_email(sender_email, sender_password, recipient_email, email_content)
            st.write(send_result)
            st.session_state["is_spam_checked"] = False

else:
    st.session_state.email_subject=''
    st.session_state.email_body=''
    email_account = st.text_input("Enter your Gmail account:")
    email_password = st.text_input("Enter your Gmail password:", type="password")

    if st.button("Fetch Latest Email"):
  
        subject, body = fetch_email(email_account, email_password)
        if body:
            st.session_state.email_subject = subject
            st.session_state.email_body = body

    if st.session_state.email_body:
        st.markdown(f"""
            <div style="border: 1px solid #bdc3c7; padding: 10px; border-radius: 5px;">
                <strong>Email Subject:</strong> {st.session_state.email_subject} <br>
                <strong>Email Body:</strong> {st.session_state.email_body}
            </div>
        """, unsafe_allow_html=True)

        language = st.selectbox("Choose language for prediction:", ["English", "Vietnamese"])
        if st.button("Check Spam Email"):
            result = process_text(st.session_state.email_body, "en" if language == "English" else "vi")
            if "Ham mail" in result:
                st.success(f"✅ {result}")
            else:
                st.error(f"🚫 {result}")
