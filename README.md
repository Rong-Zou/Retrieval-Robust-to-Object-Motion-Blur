# Retrieval-Robust-to-Object-Motion-Blur


## Description
Pytorch code for paper: [Retrieval Robust to Object Motion Blur](https://arxiv.org/abs/2404.18025)

Accepted by <strong><em>ECCV 2024</em></strong>

Rong Zou, Marc Pollefeys and Denys Rozumnyi

## Installation
The code is tested with Python 3.8.16. 

Install this repository using the following commands:

```sh
# Clone the repository
git clone https://github.com/Rong-Zou/Retrieval-Robust-to-Object-Motion-Blur.git

# Change to the project directory
cd Retrieval-Robust-to-Object-Motion-Blur

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation
Download dataset zips from <a style="color: red;">TODO</a>, and extract the data. 

You may use the following commands:

```sh
# Download the dataset
wget [TODO: data link]

# Unzip the dataset to ./data/, change the target path to your desired directory
unzip xxx.zip -d data/
```

After extraction, the data strcture should look like this:

```
parent_folder
└── real_data
    └── blur_level_1
        ├── ins1_traj1_img1.png
        ├── ins1_traj1_img2.png
        ├── ...
        └── insN_trajM_imgX.png
    ├── ...
    ├── blur_level_6
    └── stats

parent_folder
└── synthetic_data (or synthetic_data_distractors)
    └── category_1
        └── instance_1
            └── trajectory_1
                └── image_1.png
                ├── ...
                └── image_11.png
            ├── ...
            └── trajectory_T
        ├── ...
        └── instance_I
    ├── ...
    ├── category_C
    └── stats
```
The stats folder in each dataset contains pre-computed statistics that facilitate quick access to essential information of that dataset.

## Pretrained Model

Download our pre-trained model from [this link](results/train_results/link.txt).

For testing, put the model in the directory same as the link file. 

You can use the following command:

```sh
# Download the dataset
wget -P ./results/train_results https://drive.google.com/file/d/1esPk_lFtUVwJ_SCxRIGH9PsFtaI3t7SO/view?usp=sharing
```

## Testing

First modify the parameter values in the [`set_data_dirs.py`](code/set_data_dirs.py) script to configure the correct directories.

Test the model by running the testing script [`test.py`](code/test.py):
  
  ```bash
  python3 test.py
  ```

## License

This project is licensed under the [MIT License](LICENSE).

## Citation

```
@inproceedings{defmo,
  author = {Rong Zou and Marc Pollefeys and Denys Rozumnyi},
  title = {Retrieval Robust to Object Motion Blur},
  booktitle = {ECCV},
  year = {2024}
}
```