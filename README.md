```
# Curated Breast Imaging DDSM Classification with Flax NNX

This repository demonstrates how to train a Convolutional Neural Network (CNN) to classify images from the Curated Breast Imaging DDSM (CBIS-DDSM) dataset using Flax NNX. 

## Overview

The code trains a simple CNN model to classify breast density from mammogram images. It utilizes:

- Flax NNX for defining and training the model.
- Optax for optimization.
- TensorFlow Datasets (TFDS) for loading the CBIS-DDSM dataset.
- Orbax for checkpointing the model.

## Requirements

- Python 3.8 or higher
- Flax
- JAX
- Optax
- TensorFlow Datasets
- Orbax
- Typer

You can install the required packages using pip:

```bash
uv sync
```

## Dataset

The code expects the CBIS-DDSM dataset in TFRecords format. You need to download the dataset and specify its path using the `--tfrecords` option when running the script.

## Usage

To train the model, run the following command:

```bash
python main.py --train-steps 1200 --eval-every 200 --batch-size 32 --learning-rate 0.001 --momentum 0.9 --output ./output --tfrecords /path/to/cbis-ddsm.tar.gz
```

- `--train-steps`: Number of training steps.
- `--eval-every`: Evaluate the model after every N steps.
- `--batch-size`: Batch size for training.
- `--learning-rate`: Learning rate for the optimizer.
- `--momentum`: Momentum for the optimizer.
- `--output`: Output directory to save the trained model.
- `--tfrecords`: Path to the CBIS-DDSM dataset in TFRecords format.

## Model

The CNN model consists of two convolutional layers followed by two fully connected layers. The model architecture can be found in the `CNN` class within the `main.py` file.

## Training

The code trains the model using the AdamW optimizer. It computes the loss using softmax cross-entropy and updates the model parameters using gradient descent. 

## Evaluation

The model is evaluated on the test set after every `--eval-every` steps. The code computes the loss and accuracy on the test set and prints the results to the console.

## Checkpointing

The trained model is saved using Orbax. The checkpoint is saved to the directory specified by the `--output` option.

## Questions

David Cardozo <david.cardozo at me>
```