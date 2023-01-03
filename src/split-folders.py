import splitfolders

splitfolders.ratio(
    "/content/drive/MyDrive/training_data",
    output = "/content/drive/MyDrive/dataset_split",
    seed = 42,
    ratio = (.85, .15)
)