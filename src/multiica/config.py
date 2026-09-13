from pathlib import Path

####################################
# Data
####################################

dataset_dir = {
    "reach": Path("/mnt/ssd/mpotto/datasets/reach/"),
    "behavior_no_stim": Path("/Users/noahstanis/Documents/Yazdan Lab/DATA/IOS data for sharing/"),
    "behabior_stim": Path("/Users/noahstanis/Documents/Yazdan Lab/DATA/IOS data for sharing/"),
}

dataset_paths = {
    "reach": Path("/mnt/ssd/mpotto/datasets/reach/x.npy"),
    "behavior_nostim": Path("/Users/noahstanis/Documents/Yazdan Lab/DATA/IOS data for sharing/H_32ch_behavior_nostim.pkl"),
    "behavior_stim": Path("/Users/noahstanis/Documents/Yazdan Lab/DATA/IOS data for sharing/H_data_segs_all_dates.pkl"),
}

####################################
# Embeddings
####################################

####################################
# Models
####################################

model_dir = {
    "behavior_nostim": Path("/Users/noahstanis/Documents/Yazdan Lab/models/multiICA_models"),
}
