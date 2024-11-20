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


# Xây dựng giao diện với Streamlit
st.title("Email Spam Checker")

# Lưu trạng thái trong session_state
if "email_body" not in st.session_state:
    st.session_state.email_body = ""
if "email_subject" not in st.session_state:
    st.session_state.email_subject = ""

option = st.selectbox("Choose an option:", ["Write Email", "Get Email from My Gmail"])

if option == "Write Email":
    email_content = st.text_area("Enter email content:", "")
    language = st.selectbox("Choose language:", ["en", "vi"])
    if st.button("Check Spam"):
        result = process_text(email_content, language)
        if "Ham mail" in result:
            st.success(result)
            with st.form("send_email_form"):
                sender_email = st.text_input("Your email:")
                sender_password = st.text_input("Your password:", type="password")
                recipient_email = st.text_input("Recipient's email:")
                submit_button = st.form_submit_button("Send Email")
                if submit_button:
                    send_result = send_email(sender_email, sender_password, recipient_email, email_content)
                    st.success(send_result)

else:
    email_account = st.text_input("Enter your Gmail account:")
    email_password = st.text_input("Enter your Gmail password:", type="password")
    if st.button("Fetch Latest Email"):
        subject, body = fetch_email(email_account, email_password)
        if body:
            st.session_state.email_subject = subject
            st.session_state.email_body = body
    # Hiển thị email sau khi tải
    if st.session_state.email_body:
        st.write("Email Subject:", st.session_state.email_subject)
        st.text_area("Email Content:", st.session_state.email_body, key="email_display")
        language = st.selectbox("Choose language for prediction:", ["en", "vi"])
        if st.button("Check Spam Email"):
            result = process_text(st.session_state.email_body, language)
            st.write("Result:", result)
