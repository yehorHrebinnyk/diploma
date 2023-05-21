def get_unique_classes():
    from glob import glob
    from tqdm import tqdm
    import json
    unique_classes = []

    for ann in tqdm(glob("./train/annos/*") + glob("./validation/annos/*")):
        with open(ann) as f:
            data = json.load(f)

        i = 1
        while f"item{i}" in data:
            item = data[f"item{i}"]
            if item["category_name"] not in unique_classes:
                unique_classes.append(item["category_name"])
            i += 1

    unique_classes = sorted(unique_classes)
    return unique_classes


if __name__ == "__main__":
    classes = get_unique_classes()
    with open("./classes.txt", "w+") as f:
        f.write(str(classes))

    print(f"AMOUNT OF UNIQUE CLASSES = {len(classes)}")
    print(f"UNIQUE CLASSES = {classes}")
