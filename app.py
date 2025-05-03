import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load pre-trained tokenizer and model from Hugging Face
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"  # or use "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

model.eval()

# UI
st.title("Fake News Classifier")
st.write("Enter a news statement, and we'll tell you whether it's likely **True** or **False**.")

user_input = st.text_area("Enter news text:", height=150)

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=256)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            label = torch.argmax(probs, dim=1).item()
            confidence = probs[0][label].item()

        if label == 1:
            st.success(f"🟢 Likely **TRUE** with confidence {confidence:.2f}")
        else:
            st.error(f"🔴 Likely **FALSE** with confidence {confidence:.2f}")



