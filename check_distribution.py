import json
import argparse
import matplotlib.pyplot as plt

from tqdm import tqdm
from glob import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default='train')
    args = parser.parse_args()
    folder = args.dir
    dist = {}

    for ann_file in tqdm(glob(f"./{folder}/annos/*.json")):

        with open(ann_file) as f:
            data = json.load(f)

        for k in data.keys():
            item = data[k]

            if type(item) is not dict:
                continue

            category_name = item["category_name"]

            if category_name not in dist:
                dist[category_name] = 0

            dist[category_name] += 1

    classes = list(dist.keys())
    values = list(dist.values())

    plt.bar(range(len(classes)), values, color='teal')
    plt.xticks(range(len(classes)), classes, rotation=90)
    plt.title(f"{folder.upper()} dataset")
    plt.show()
