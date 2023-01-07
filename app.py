import fastapi
import pickle
import string
import nltk

nltk.download("stopwords")
nltk.download('punkt')

from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from pydantic import BaseModel



# Load the pre-trained machine learning model and feature vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Define a function to preprocess text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
      ps = PorterStemmer()
      y.append(ps.stem(i))
    return " ".join(y)

# Define a model for the request payload
class TextClassificationRequest(BaseModel):
  text: str

# Define a model for the response payload
class TextClassificationResponse(BaseModel):
  label: str

# Create an instance of FastAPI
app = fastapi.FastAPI()

print(type(app))

@app.get("/")
async def hello():
  return {"hello": "world"}

# Define an endpoint for classifying text messages
@app.post('/classify')
async def classify(request: TextClassificationRequest):
  # Preprocess the text message
  transformed_sms = transform_text(request.text)
  # transformed_sms = request.text
  # print(transformed_sms)
  # return {"hello": "world23456"}
  # Vectorize the text message
  vector_input = vectorizer.transform([transformed_sms])
  # Predict whether the text message is spam or not
  result = model.predict(vector_input)[0]
  # Return the classification label
  print("Result:", result)

  if result == 1:
    return TextClassificationResponse(label='spam')
  else:
    return TextClassificationResponse(label='not spam')
