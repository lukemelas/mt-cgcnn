# Bayesian Graph Convolutional Neural Networks for Crystallography 

This repository contains work on Bayesian GCNs for crystallography. The goal of the project is to infer material properties with uncertainty measurements from material properties. We build upon [CGCNN](https://arxiv.org/abs/1710.10324) and [MT-CGCNN](https://arxiv.org/abs/1811.05660). 

## Table of Contents

- [Prerequisites](#prerequisites)
- [Usage](#usage)
  - [Get data](#download-cif-files-from-the-materials-project)
  - [Training](#trainingl)
- [License](#license)

##  Prerequisites

```bash
pip install -r requirements.txt
```

## Usage

### Download CIF files from the Materials Project

To download CIF files, you will need to first [generate an API KEY]. Then run:

```bash
python utils.py --api_key API_KEY --mp_ids csvs/mpids_full.csv --output data/materials_project
```

### Training

To see training of an instance of MT-CGCNN, you can just run the following:

```bash
python init.py
```
This will run a demo using datasource: `data/sample/` and some pre-defined set of parameters. But, you can change and play with the parameters of the model using the `init.py` file. To run multiple iterations of the same experiment (one motivation can be to get average error), you can run the following code:

```bash
python init.py --idx 1
```

To reproduce results stated in the paper, you might need to tune the parameters in the hyperparameter space mentioned in the paper. Also, average MAE is reported for 5 runs of the experiment.

After training, you will get multiple files in `results` folder present within the datasource (For eg., for demo case results will be saved in `data/sample/results/0/`). The most important ones are:

- `model_best.pth.tar`: stores the MT-CGCNN model with the best validation accuracy.
- `checkpoint.pth.tar`: stores the MT-CGCNN model at the last epoch.
- `test_results.csv`: stores the `ID`, target values, and predicted values for each crystal in test set.
- `logfile.log`: A complete log of the experiment (useful for DEBUGGING purposes)

The other files are useful to understand how well-trained the model is and can be referred for DEBUGGING purposes. Briefly, the files contain information about the training & validation losses, training & validation errors and also some useful plots.

## License

MT-CGCNN is released under the MIT License.
