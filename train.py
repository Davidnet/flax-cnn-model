import tarfile
from functools import partial
from pathlib import Path

import jax.numpy as jnp  # JAX NumPy
import optax
import orbax.checkpoint as orbax
import tensorflow as tf
import tensorflow_datasets as tfds
import typer
from flax import nnx  # Flax NNX API


def main(
    train_steps: int = typer.Option(..., help="Number of training steps"),
    eval_every: int = typer.Option(..., help="Evaluate after every N steps"),
    batch_size: int = typer.Option(..., help="Batch size for training"),
    learning_rate: float = typer.Option(..., help="Learning rate for optimizer"),
    momentum: float = typer.Option(..., help="Momentum for optimizer"),
    output: Path = typer.Option(..., help="Output directory for the results"),
    tfrecords: Path = typer.Option(..., help="Path to the TFRecords dataset"),
):
    curated_breast_imaging_ddsm = (
        Path.home() / "tensorflow_datasets/curated_breast_imaging_ddsm"
    )
    curated_breast_imaging_ddsm.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tfrecords) as tar:
        tar.extractall(curated_breast_imaging_ddsm)

    train_ds = tfds.load("curated_breast_imaging_ddsm", split="train")
    test_ds = tfds.load("curated_breast_imaging_ddsm", split="test")

    tf.random.set_seed(0)  # set random seed for reproducibility

    train_steps = 1200
    eval_every = 200
    batch_size = 32

    train_ds = train_ds.map(
        lambda sample: {
            "image": tf.cast(sample["image"], tf.float32) / 255,
            "label": sample["label"],
        }
    )  # normalize train set
    test_ds = test_ds.map(
        lambda sample: {
            "image": tf.cast(sample["image"], tf.float32) / 255,
            "label": sample["label"],
        }
    )  # normalize test set

    # create shuffled dataset by allocating a buffer size of 1024 to randomly draw elements from
    train_ds = train_ds.repeat().shuffle(49780)
    # group into batches of batch_size and skip incomplete batch, prefetch the next sample to improve latency
    train_ds = (
        train_ds.batch(batch_size, drop_remainder=True).take(train_steps).prefetch(1)
    )
    # group into batches of batch_size and skip incomplete batch, prefetch the next sample to improve latency
    test_ds = test_ds.batch(batch_size, drop_remainder=True).prefetch(1)

    class CNN(nnx.Module):
        """A simple CNN model."""

        def __init__(self, *, rngs: nnx.Rngs):
            self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), rngs=rngs)
            self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
            self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))
            self.linear1 = nnx.Linear(200704, 256, rngs=rngs)
            self.linear2 = nnx.Linear(256, 5, rngs=rngs)

        def __call__(self, x):
            x = self.avg_pool(nnx.relu(self.conv1(x)))
            x = self.avg_pool(nnx.relu(self.conv2(x)))
            x = x.reshape(x.shape[0], -1)  # flatten
            x = nnx.relu(self.linear1(x))
            x = self.linear2(x)
            return x

    model = CNN(rngs=nnx.Rngs(0))

    # warm-up the model
    y = model(jnp.ones((1, 224, 224, 1)))

    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate, momentum))
    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average("loss"),
    )

    def loss_fn(model: CNN, batch):
        logits = model(batch["image"])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch["label"]
        ).mean()
        return loss, logits

    @nnx.jit
    def train_step(
        model: CNN, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch
    ):
        """Train for a single step."""
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(model, batch)
        metrics.update(
            loss=loss, logits=logits, labels=batch["label"]
        )  # inplace updates
        optimizer.update(grads)  # inplace updates

    @nnx.jit
    def eval_step(model: CNN, metrics: nnx.MultiMetric, batch):
        loss, logits = loss_fn(model, batch)
        metrics.update(
            loss=loss, logits=logits, labels=batch["label"]
        )  # inplace updates

    metrics_history = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }

    for step, batch in enumerate(train_ds.as_numpy_iterator()):
        # Run the optimization for one step and make a stateful update to the following:
        # - the train state's model parameters
        # - the optimizer state
        # - the training loss and accuracy batch metrics
        train_step(model, optimizer, metrics, batch)

        if step > 0 and (
            step % eval_every == 0 or step == train_steps - 1
        ):  # one training epoch has passed
            # Log training metrics
            for metric, value in metrics.compute().items():  # compute metrics
                metrics_history[f"train_{metric}"].append(value)  # record metrics
            metrics.reset()  # reset metrics for test set

            # Compute metrics on the test set after each training epoch
            for test_batch in test_ds.as_numpy_iterator():
                eval_step(model, metrics, test_batch)

            # Log test metrics
            for metric, value in metrics.compute().items():
                metrics_history[f"test_{metric}"].append(value)
            metrics.reset()  # reset metrics for next training epoch

            print(
                f"[train] step: {step}, "
                f"loss: {metrics_history['train_loss'][-1]}, "
                f"accuracy: {metrics_history['train_accuracy'][-1] * 100}"
            )
            print(
                f"[test] step: {step}, "
                f"loss: {metrics_history['test_loss'][-1]}, "
                f"accuracy: {metrics_history['test_accuracy'][-1] * 100}"
            )

    @nnx.jit
    def pred_step(model: CNN, batch):
        logits = model(batch["image"])
        return logits.argmax(axis=1)

    test_batch = test_ds.as_numpy_iterator().next()
    pred = pred_step(model, test_batch)

    print(f"Predictions: {pred}")

    checkpointer = orbax.PyTreeCheckpointer()

    output.mkdir(parents=True, exist_ok=True)
    checkpointer.save(output.absolute().as_posix(), nnx.state(model), force=True)


if __name__ == "__main__":
    typer.run(main)
