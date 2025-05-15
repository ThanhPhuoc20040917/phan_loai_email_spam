import os
import re
import pandas as pd
import tkinter as tk
from tkinter import scrolledtext, messagebox
import subprocess
import sys
import imaplib
import email
from email.header import decode_header
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
except ImportError:
    install_package("nltk")
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    word_tokenize("example")
except LookupError:
    nltk.download('punkt')

stop_words = set(stopwords.words('english'))
vectorizer = None
classifier = None
data_loaded = False 

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)  
    text = re.sub(r"[^a-z0-9\s]", " ", text) 
    return text.strip()

def load_data():
    global vectorizer, classifier, data_loaded
    try:
        file_path = os.path.join(os.path.dirname(__file__), "messages.csv")
        df = pd.read_csv(file_path)
        
        if "subject" not in df.columns or "message" not in df.columns or "label" not in df.columns:
            messagebox.showerror("Lỗi", "File messages.csv không có đúng định dạng (cột subject, message, label).")
            return
        
        df["subject"] = df["subject"].fillna("")
        df["message"] = df["message"].fillna("")
        
        df["text"] = (df["subject"] + " " + df["message"]).apply(clean_text)
        X = df["text"]
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        vectorizer = TfidfVectorizer(stop_words='english')
        X_train_vectorized = vectorizer.fit_transform(X_train)
        classifier = MultinomialNB()
        classifier.fit(X_train_vectorized, y_train)
        X_test_vectorized = vectorizer.transform(X_test)
        y_pred = classifier.predict(X_test_vectorized)
        accuracy = accuracy_score(y_test, y_pred)
        messagebox.showinfo("Huấn luyện xong", f"Huấn luyện hoàn tất!\nĐộ chính xác trên tập kiểm tra: {accuracy:.2f}")
        data_loaded = True 
    except Exception as e:
        messagebox.showerror("Lỗi", f"Không thể tải dữ liệu: {str(e)}")

def connect_to_gmail(email_address, app_password):
    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(email_address, app_password)
        mail.select("INBOX")
        return mail
    except Exception as e:
        messagebox.showerror("Lỗi", f"Không thể kết nối Gmail: {str(e)}")
        return None

def fetch_emails(mail, num_emails=50):
    try:
        status, messages = mail.search(None, "ALL")
        email_ids = messages[0].split()[-num_emails:]
        emails = []
        for email_id in email_ids:
            status, msg_data = mail.fetch(email_id, "(RFC822)")
            msg = email.message_from_bytes(msg_data[0][1])
            subject, encoding = decode_header(msg["subject"])[0]
            if isinstance(subject, bytes):
                subject = subject.decode(encoding if encoding else "utf-8")
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_payload(decode=True).decode()
                        break
            else:
                body = msg.get_payload(decode=True).decode()
            emails.append((subject, body))
        return emails
    except Exception as e:
        messagebox.showerror("Lỗi", f"Không thể lấy email: {str(e)}")
        return []

def classify_gmail_emails():
    if not data_loaded:
        messagebox.showwarning("Cảnh báo", "Dữ liệu chưa được tải! Vui lòng huấn luyện mô hình trước.")
        return
    
    email_address = email_entry.get().strip()
    app_password = password_entry.get().strip()
    
    if not email_address or not app_password:
        messagebox.showwarning("Cảnh báo", "Vui lòng nhập thông tin Gmail.")
        return
    
    mail = connect_to_gmail(email_address, app_password)
    if not mail:
        return
    
    emails = fetch_emails(mail, num_emails=50)
    text_area.delete("1.0", tk.END)
    
    for i, (subject, body) in enumerate(emails, 1):
        email_text_cleaned = clean_text(subject + " " + body)
        email_vectorized = vectorizer.transform([email_text_cleaned])
        prediction = classifier.predict(email_vectorized)[0]
        label_text = "Thư rác" if prediction == 1 else "Không phải thư rác"
        text_area.insert(tk.END, f"Email {i} - Tiêu đề: {subject}\n")
        text_area.insert(tk.END, f"Dự đoán: {label_text}\n")
        text_area.insert(tk.END, "-" * 60 + "\n")
    
    mail.logout()
    text_area.insert(tk.END, "\nĐã hiển thị 50 email từ Gmail.\n")

window = tk.Tk()
window.title("Phân loại Email Thư Rác từ Gmail (Naive Bayes)")

tk.Button(window, text="Tải dữ liệu & Huấn luyện mô hình", command=load_data).pack(pady=5)

tk.Label(window, text="Địa chỉ Gmail:").pack(pady=5)
email_entry = tk.Entry(window, width=50)
email_entry.pack()

tk.Label(window, text="Mật khẩu ứng dụng:").pack(pady=5)
password_entry = tk.Entry(window, width=50)
password_entry.pack()
password_entry.config(show='*')

tk.Checkbutton(window, text="Hiện mật khẩu", command=lambda: password_entry.config(show='' if password_entry.cget('show')=='*' else '*')).pack(pady=2)

tk.Button(window, text="Phân loại Email từ Gmail", command=classify_gmail_emails).pack(pady=10)

text_area = scrolledtext.ScrolledText(window, width=80, height=30)
text_area.pack(pady=10)

window.mainloop()
