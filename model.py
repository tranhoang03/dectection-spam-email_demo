import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # Thư viện để lưu và tải mô hình
from sklearn.feature_extraction.text import TfidfVectorizer
import standard_data



# Hàm hiển thị ma trận nhầm lẫn
def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def process(input, language):
    data = input

    # Visualization dữ liệu
    print("Visualization số lượng mẫu trong mỗi lớp của tập dữ liệu tiếng " + language + ' :')
    data['spam'].replace({0: 'ham', 1: 'spam'}, inplace=True)
    data['spam'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title('Số lượng mẫu trong mỗi lớp')
    plt.xlabel('Class')
    plt.ylabel('Số lượng')
    plt.show()

    # Downsampling để cân bằng
    print("Downsampling")
    ham_msg = data[data.spam == 'ham']
    spam_msg = data[data.spam == 'spam']
    ham_msg = ham_msg.sample(n=len(spam_msg), random_state=42)

    balanced_data = pd.concat([ham_msg, spam_msg], ignore_index=True)

    # Tiền xử lý dữ liệu
    print("Tiền xử lý dữ liệu")
    if language == "vi":
        balanced_data['text'] = balanced_data['text'].apply(lambda text: standard_data.standard_vi(text))
    else:
        balanced_data['text'] = balanced_data['text'].apply(lambda text: standard_data.standard_en(text))

    # Visualization số lượng từ trong mỗi email sau khi tiền xử lý
    print("Visualization số lượng từ trong mỗi email")
    word_counts = balanced_data['text'].apply(lambda x: len(x.split()))
    plt.hist(word_counts, bins=30, color='purple', edgecolor='black')
    plt.title('Số lượng từ trong mỗi email')
    plt.xlabel('Số từ')
    plt.ylabel('Số lượng email')
    plt.show()

    # Chia dữ liệu
    print("Chia dữ liệu")
    train_X, test_X, train_Y, test_Y = train_test_split(balanced_data['text'],
                                                        balanced_data['spam'],
                                                        test_size=0.2,
                                                        random_state=42)

    # Sử dụng TF-IDF để chuyển đổi văn bản thành vector đặc trưng
    print("Chuyển đổi văn bản thành vector TF-IDF")
    vectorizer = TfidfVectorizer()
    train_X = vectorizer.fit_transform(train_X)
    test_X = vectorizer.transform(test_X)

    # Huấn luyện với SVM
    print("Huấn luyện SVM")
    model = SVC(kernel='linear', C=1, probability=True)
    model.fit(train_X, train_Y)

    # Lưu model và vectorizer
    joblib.dump(model, 'svm_model_' + language + '.joblib')
    joblib.dump(vectorizer, 'vectorizer_' + language + '.joblib')

    # Đánh giá
    print("Đánh giá")
    test_predictions = model.predict(test_X)
    test_accuracy = accuracy_score(test_Y, test_predictions)
    print('Test Accuracy :', test_accuracy)
    print(classification_report(test_Y, test_predictions))

    # Ma trận nhầm lẫn
    plot_confusion_matrix(test_Y, test_predictions, labels=['ham', 'spam'])

    return model, vectorizer


