import spacy
import json
nlp = spacy.load('en')
json_data = open('Book_Data_Set.json',encoding="utf8")
data = json.load(json_data)
title = ""
for i in range (0, 10):
	doc = nlp(data[i]['description'])
	for ent in doc.ents:
		print(ent.text,ent.label_)
	print("**************")