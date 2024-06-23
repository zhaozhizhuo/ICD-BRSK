from simcse import SimCSE

model = SimCSE("./sup-simcse-bert-base-uncased")

sentences_a = ['Screening examination for other specified bacterial and spirochetal']
sentences_b = ['Acute paralytic poliomyelitis specified as bulbar, poliovirus type III']
similarities = model.similarity(sentences_a, sentences_b)

print(similarities)