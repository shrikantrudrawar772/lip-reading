import os

root = "D:/grid_word_dataset/"

for label in ["bin", "lay", "place", "set"]:
    folder = os.path.join(root, label)
    print(label, len(os.listdir(folder)))
