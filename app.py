from fastapi import FastAPI,Query
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pydantic import BaseModel
app = FastAPI()
class TextInput(BaseModel):
    text :str
sia = SentimentIntensityAnalyzer()
@app.post("/sentiment")
def sentimentAnalysis(data:TextInput):
    score = sia.polarity_scores(data.text)
    compound = score["compound"]
    if compound <= -0.05:
        label = "Negative"
    elif compound >= 0.05:
        label = "Positive"
    else:
        label = "Neutral"
    return { "text": data.text,
        "sentiment": label,
        "compound": compound,
        "score": score
    }
