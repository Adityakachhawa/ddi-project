import pickle

interaction_list = pickle.load(open("resources/interaction_list.pkl", "rb"))  # Corrected folder name
interaction_mapping = pickle.load(open("resources/interaction_mapping.pkl", "rb"))

for idx, text in enumerate(interaction_list):
    assert interaction_mapping[text] == idx, "Index mismatch!"
print("✅ All interactions properly ordered!")
