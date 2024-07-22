# ROMB Dataset: Dataset for Retrieval with Object Motion Blur

## Dataset Description

### Overview
The ROMB dataset includes real and synthetic data designed for retrieval in the challenging presence of object motion blur. The images contained in the ROMB dataset share the following attributes: 
- Image Type: png
- Width:      320 pixels
- Height:     240 pixels

### File Contents

- `real_data.zip`: This novel real-world dataset is specifically constructed to test the methods' effectiveness in real-world scenarios. This dataset contains 13,093 real images for 35 different objects, with an average of 374 images for each object. We move each object 3 to 5 times along different trajectories in front of the camera. The images are generated from 139 trajectories recorded using a GoPro 7 Hero camera at 240 fps with full exposure. For each object, we use the trajectory with the least number of generated images as query, and the rest is used as database. None of the real-world objects are in synthetic data and 2/3 of which are from new categories not present in synthetic data. We manually annotate the data, assigning each image to one of 6 blur levels based on the perceived blur, and we use the ratio of the trajectory length to the average object size as a reference. The annotation enables a more comprehensive analysis and evaluation of model performance.

- `synthetic_data.zip`: This dataset is designed to train and evaluate methods for retrieval with object motion blur. It includes 1.5 million images for 1138 different objects (train:val:test = 792:153:193) from 39 categories. The objects are chosen from the [ShapeNet](https://shapenet.org/) dataset. The images in this dataset contain 3D objects moving along different trajectories in front of different backgrounds sampled from the [LHQ](https://github.com/universome/alis) dataset. For each object, we generate 120 random trajectories and render 1 sharp and 10 motion-blurred images with varying degrees of blur for each trajectory. Images rendered from each trajectory have the same background. For the training objects, we take one image for each trajectory while ensuring a balanced distribution of images across different blur levels. For each of the test objects, we sample 20 trajectories to form a query set, and the rest is used for the database. During evaluation, the unique mapping between trajectory and background enables invesgating the impact of varying levels of motion blur on retrieval performance while eliminating the influence of the background.

- `synthetic_data_distractors.zip`: This dataset contains over 1 million images acting as distractors used to evaluate methods in a more challenging setup. We use the same object categories as before to increase the difficulty of the distractors in terms of intra-class similarity. We randomly select 40 objects per category, excluding the 30 previously chosen objects, and generate 70 trajectories for each. For each trajectory, similar to before, we render one sharp image and 10 blurry images with varying degrees of blur. Backgrounds are also selected from a different subset of LHQ. To increase the difficulty and variety, each image in this dataset has a unique background.

## Usage Instructions
### Download and Extraction
Download the zip files from this [link](https://cvg-data.inf.ethz.ch/romb/) and extract the contents using any standard unzip tool.

Alternatively, you may use the following commands:

```sh
# Download the dataset
wget https://cvg-data.inf.ethz.ch/romb/real_data.zip
wget https://cvg-data.inf.ethz.ch/romb/synthetic_data.zip
wget https://cvg-data.inf.ethz.ch/romb/synthetic_data_distractors.zip

# Unzip the dataset to ./data/, change the target path to your desired directory
unzip real_data.zip -d data/
unzip synthetic_data.zip -d data/
unzip synthetic_data_distractors.zip -d data/
```

### Usage
Check out our [github repository](https://github.com/Rong-Zou/Retrieval-Robust-to-Object-Motion-Blur) for code of using the data.

## Data Structure

After extraction, the data strcture should look like this:
### real_data.zip
```
parent_folder
└── real_data
    └── blur_level_1
        ├── obj1_traj1_img1.png
        ├── obj1_traj1_img2.png
        ├── ...
        └── objN_trajM_imgX.png
    ├── ...
    ├── blur_level_6
    └── stats
```
### synthetic_data.zip
```
parent_folder
└── synthetic_data
    └── category_1
        └── object_1
            └── trajectory_1
                └── image_1.png
                ├── ...
                └── image_11.png
            ├── ...
            └── trajectory_120
        ├── ...
        └── object_30
    ├── ...
    ├── category_39
    └── stats
```
### synthetic_data_distractors.zip
```
parent_folder
└── synthetic_data_distractors
    └── category_1
        └── object_1
            └── trajectory_1
                └── image_1.png
                ├── ...
                └── image_11.png
            ├── ...
            └── trajectory_70
        ├── ...
        └── object_40
    ├── ...
    ├── category_39
    └── stats
```

## Pre-computed Statistics

The stats folder in each dataset contains pre-computed statistics that facilitate quick access to essential information of that dataset. The information contained in the stats folders are described as follows:

### Real Data
- `all_images.json`: paths of all real images
- `database.npy`: a list of (path, blur level, object label) for each image in the database
- `queries.npy`: a list of (path, blur level, object label) for each query image

### Synthetic Data
- `blur_stats`:
    - `img_paths_by_blurlevel.json`: a list of sub-lists, each sub-list contain paths of all images at the corresponding blur level 
    - `blur_values_by_blurlevel.npz`: a list of sub-lists, each sub-list contain blur values of all images at the corresponding blur level, each blur value corresponds to the image at the same index in `img_paths_by_blurlevel.json` (blur value = 1 - blur severity, for definition of blur severity and blur level, please refer to our [paper](#citation))
- `loader`:
    - `data_split_info.json`: information about train/val/test split of data, including the number of objects in each split and a detailed list of objects in each slpit. It also contains the correspondences between category IDs and category names
    - `loader folder for each split`: pre-computed blur values and blur levels for images of each object in each split
- `traj_stats`: intermediate results, statistics of the trajectories of each object, used to make the number of images with different blur levels as even as possible for each object in the database and query

### Synthetic Data Distractors
- `img_paths.json`: paths of all distractor images
- `blur_levels.npy`: a list of blur levels corresponding to each image in `img_paths.json`
- `blur_vals.npy`: a list of blur values corresponding to each image in `img_paths.json`


To use the pre-calculated information of the datasets, check out our [github repository](https://github.com/Rong-Zou/Retrieval-Robust-to-Object-Motion-Blur) for the code.

## License
This dataset is licensed under the MIT License. 

You can read the full terms of the license at the following [link](https://github.com/Rong-Zou/Retrieval-Robust-to-Object-Motion-Blur/blob/main/LICENSE).

## Citation
If you use our dataset in your research or project, please cite it as follows:
```
@inproceedings{blur_retrieval,
  author = {Rong Zou and Marc Pollefeys and Denys Rozumnyi},
  title = {Retrieval Robust to Object Motion Blur},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year = {2024}
}
```

Thank you for using the ROMB Dataset!
