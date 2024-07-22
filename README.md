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
Download dataset zips from [this link](https://cvg-data.inf.ethz.ch/romb), and extract the data. 

You may use the following commands:

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

See the [dataset page](data/data.md) for more details.

## Pretrained Model

Download our pre-trained model from [this link](results/train_results/link.txt).

For testing, put the model in the directory same as the link file. 

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
@inproceedings{blur_retrieval,
  author = {Rong Zou and Marc Pollefeys and Denys Rozumnyi},
  title = {Retrieval Robust to Object Motion Blur},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year = {2024}
}
```