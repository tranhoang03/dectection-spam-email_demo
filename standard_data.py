import pandas as pd
import prepare_stopwords

stop_words_vi = prepare_stopwords.get_vie_stopwords()
stop_words_eng = prepare_stopwords.get_eng_stopwords()


def standard_vi(input_text):
    input_text = input_text.lower()
    
    # Xóa từ 'chủ đề:'
    input_text = input_text.replace('chủ đề:', '')  # Sửa đổi ở đây

    # Loại bỏ dấu câu
    punctuations = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    for char in punctuations:
        input_text = input_text.replace(char, '')
        
    # Loại bỏ dấu cách
    input_text = " ".join(input_text.split())
    
    # Xóa stopword
    input_text = " " + input_text + " "
    delete = [0] * len(input_text)
    
    for word in stop_words_vi:
        word = " " + word + " "
        index = input_text.find(word)
        while index != -1:
            for i in range(index + 1, index + len(word) - 1):
                delete[i] = 1
            index = input_text.find(word, index + 1)
    
    output = "".join([input_text[i] if delete[i] == 0 else "" for i in range(len(input_text))])
    output = " ".join(output.split())
    return output

def standard_en(input_text):
    input_text = input_text.lower()
    
    # Xóa từ 'subject'
    input_text = input_text.replace('subject', '')  # Sửa đổi ở đây

    # Loại bỏ dấu câu
    punctuations = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    for char in punctuations:
        input_text = input_text.replace(char, '')
    
    imp_words = []
    for word in input_text.split():
        if word not in stop_words_eng:
            imp_words.append(word)
            
    output = " ".join(imp_words)
    return output

# Test hàm
balanced_data = pd.DataFrame({'text': ['Chủ đề: Xin chào bạn!', 'Subject: Introduction']})
balanced_data['text'] = balanced_data['text'].apply(lambda x: standard_vi(x) if x.startswith('Chủ đề:') else standard_en(x))
print(balanced_data)
