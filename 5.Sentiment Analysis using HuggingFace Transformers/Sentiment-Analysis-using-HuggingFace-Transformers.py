from transformers import pipeline

# Load pre-trained sentiment-analysis pipeline
classifier = pipeline("sentiment-analysis")

# Input sentence
sentence = "Despite the high price, the performance of the new MacBook is outstanding."

# Analyze sentiment
result = classifier(sentence)[0]

# Print result
print(f"Sentiment: {result['label']}")
print(f"Confidence Score: {round(result['score'], 4)}")
