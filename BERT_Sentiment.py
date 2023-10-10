import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
from docx import Document
import re
from keywords import keywords

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

# Load the document
document = Document("equifax_cc.docx")
text = " ".join([paragraph.text for paragraph in document.paragraphs])


# Initialize the sentiment analysis pipeline
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Create a dictionary to store paragraphs with their corresponding keywords and sentiment
paragraphs_with_keywords = {}

# Split the text into smaller segments (e.g., sentences)
segments = re.split(r'(?<=[.!?])', text)

for segment in segments:
    for keyword in keywords:
        if keyword in segment:
            # Perform sentiment analysis on the segment
            segment_results = nlp(segment)

            # Save the segment, keywords found, and sentiment to the dictionary
            if keyword not in paragraphs_with_keywords:
                paragraphs_with_keywords[keyword] = []
            paragraphs_with_keywords[keyword].append({"paragraph": segment, "sentiment": segment_results[0]})

# Print and save the results to a file
output_file = "output.txt"
with open(output_file, "w") as file:
    for keyword, paragraphs in paragraphs_with_keywords.items():
        file.write(f"Keyword: {keyword}\n")
        for paragraph in paragraphs:
            file.write(f"Paragraph:\n{paragraph['paragraph']}\n")
            file.write(f"Sentiment: {paragraph['sentiment']['label']}, Score: {paragraph['sentiment']['score']}\n\n")

print(f"Results saved to {output_file}")
