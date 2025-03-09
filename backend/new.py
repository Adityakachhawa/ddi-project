# Test script
import pickle

interaction_list = pickle.load(open("interaction_list.pkl", "rb"))
interaction_mapping = pickle.load(open("interaction_mapping.pkl", "rb"))

for idx, text in enumerate(interaction_list):
    assert interaction_mapping[text] == idx, "Index mismatch!"
print("✅ All interactions properly ordered!")