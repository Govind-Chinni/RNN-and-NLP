import spacy

# Load English model
nlp = spacy.load("en_core_web_sm")

# Input sentence
sentence = "Barack Obama served as the 44th President of the United States and won the Nobel Peace Prize in 2009."

# Process the sentence
doc = nlp(sentence)

# Extract named entities
print("Named Entities Found:\n")
for ent in doc.ents:
    print(f"Text: {ent.text}")
    print(f"Label: {ent.label_}")
    print(f"Start Char: {ent.start_char}, End Char: {ent.end_char}")
    print("---")
