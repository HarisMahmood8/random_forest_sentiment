import docx
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

clf = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

doc = docx.Document('equifax_cc.docx')

text = ''
for paragraph in doc.paragraphs:
    text += paragraph.text + ' '

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    return ' '.join(words)

preprocessed_text = preprocess_text(text)

text_tfidf = vectorizer.transform([preprocessed_text])

sentiment = clf.predict(text_tfidf)[0]

sentiment_label = "Positive" if sentiment == 1 else "Negative"

print(f"Sentiment: {sentiment_label}")
