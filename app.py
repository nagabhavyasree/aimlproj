import streamlit as st
import gdown
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Google Drive file ID
file_id = "your_file_id"  # Replace with your actual Google Drive file ID
output = "bert_model.pth"

# Download the model from Google Drive
gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.load_state_dict(torch.load("bert_model.pth", map_location=torch.device("cpu")))
model.eval()

# Streamlit UI
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


