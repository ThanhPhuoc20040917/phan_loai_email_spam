import tkinter as tk
from tkinter import scrolledtext, messagebox
import subprocess
import sys
import imaplib
import email
from email.header import decode_header
import re

# Kiểm tra và cài đặt các thư viện cần thiết
def install_package(package):
    """Cài đặt một package nếu chưa được cài đặt."""
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

# Tải dữ liệu cần thiết cho NLTK (chạy lần đầu)
try:
    stopwords.words('english')  # Vẫn sử dụng stopwords tiếng Anh vì email thường là tiếng Anh
except LookupError:
    nltk.download('stopwords')
try:
    word_tokenize("example")
except LookupError:
    nltk.download('punkt_tab')

# --- Danh sách từ khóa thư rác đơn giản ---
SPAM_KEYWORDS = {'win', 'free', 'money', 'prize', 'urgent', 'lottery', 'click', 'buy', 'offer'}
stop_words = set(stopwords.words('english'))

# --- Kết nối đến Gmail qua IMAP ---
def connect_to_gmail(email_address, app_password):
    """Kết nối đến Gmail qua IMAP."""
    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        print(f"Đang kết nối với Gmail bằng email: {email_address}")
        mail.login(email_address, app_password)
        print("Đăng nhập thành công!")
        mail.select("INBOX")  # Chọn hộp thư đến
        return mail
    except Exception as e:
        messagebox.showerror("Lỗi", f"Không thể kết nối đến Gmail: {str(e)}\nVui lòng kiểm tra email, mật khẩu ứng dụng, và đảm bảo IMAP được bật trong Gmail.")
        exit()

# --- Lấy nội dung email ---
def fetch_emails(mail, num_emails=20):
    """Lấy nội dung của một số email từ Gmail."""
    try:
        status, messages = mail.search(None, "ALL")
        email_ids = messages[0].split()[-num_emails:]  # Lấy num_emails email mới nhất

        emails = []
        for email_id in email_ids:
            status, msg_data = mail.fetch(email_id, "(RFC822)")
            msg = email.message_from_bytes(msg_data[0][1])
            
            subject, encoding = decode_header(msg["subject"])[0]
            if isinstance(subject, bytes):
                subject = subject.decode(encoding if encoding else "utf-8")
            
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_payload(decode=True).decode()
                        emails.append((subject, body))
                        break
            else:
                body = msg.get_payload(decode=True).decode()
                emails.append((subject, body))
        
        return emails
    except Exception as e:
        messagebox.showerror("Lỗi", f"Không thể lấy email: {str(e)}")
        return []

# --- Hàm dự đoán thư rác dựa trên từ khóa ---b11_detected = any(keyword in filtered_tokens for keyword in SPAM_KEYWORDS)
    print("Từ khóa thư rác phát hiện:", [keyword for keyword in filtered_tokens if keyword in SPAM_KEYWORDS])
    
    return "Thư rác" if spam_detected else "Không phải thư rác"

# --- Xây dựng giao diện người dùng ---
def classify_emails():
    email_address = email_entry.get().strip()
    app_password = password_entry.get().strip()
    
    if not email_address or not app_password:
        messagebox.showwarning("Cảnh báo", "Vui lòng nhập email và mật khẩu ứng dụng.")
        return
    
    mail = connect_to_gmail(email_address, app_password)
    
    emails = fetch_emails(mail, num_emails=20)
    
    text_area.delete("1.0", tk.END)
    
    for i, (subject, body) in enumerate(emails, 1):
        result = predict_spam(body)
        text_area.insert(tk.END, f"Email {i} - Tiêu đề: {subject}\n")
        text_area.insert(tk.END, f"Kết quả: {result}\n")
        text_area.insert(tk.END, "-" * 50 + "\n")
    
    mail.logout()

def toggle_password_visibility():
    current_show = password_entry.cget('show')
    if current_show == '*':
        password_entry.config(show='')
    else:
        password_entry.config(show='*')

# Tạo giao diện Tkinter
window = tk.Tk()
window.title("Ứng Dụng Phân Loại Email Thư Rác Từ Gmail")

# Nhập email và mật khẩu ứng dụng
email_label = tk.Label(window, text="Địa chỉ Gmail:")
email_label.pack(pady=5)
email_entry = tk.Entry(window, width=50)
email_entry.pack()

password_label = tk.Label(window, text="Mật khẩu ứng dụng:")
password_label.pack(pady=5)
password_entry = tk.Entry(window, width=50)
password_entry.pack()

# Nút hiển thị/ẩn mật khẩu
show_password_var = tk.BooleanVar()
show_password_check = tk.Checkbutton(window, text="Hiện mật khẩu", variable=show_password_var, command=toggle_password_visibility)
show_password_check.pack(pady=2)

# Nút phân loại
predict_button = tk.Button(window, text="Phân Loại Email", command=classify_emails)
predict_button.pack(pady=10)

# Khu vực hiển thị kết quả
text_area = scrolledtext.ScrolledText(window, width=60, height=20)
text_area.pack(pady=10)

# Đặt mật khẩu ẩn mặc định
password_entry.config(show='*')

window.mainloop()