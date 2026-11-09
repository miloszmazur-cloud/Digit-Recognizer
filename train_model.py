import argparse
import random
import numpy as np
from sklearn.datasets import fetch_openml
from model import AI

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs')
parser.add_argument('--seed', type=int, default=123, help='Random seed for reproducibility')
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

print("Downloading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False, parser='liac-arff')

# Normalize pixel values
X = mnist.data / 255.0
y = mnist.target.astype(int)

ai = AI()

total_images = len(X)
print(f"Starting training on {total_images} images for {args.epochs} epochs...")

for epoch in range(args.epochs):
    indices = np.random.permutation(total_images)

    for step, idx in enumerate(indices):
        if step % 2000 == 0:
            print(f"Epoch {epoch + 1}/{args.epochs} - {step}/{total_images} images...")

        ai.train(X[idx], int(y[idx]))

        # Train class 10 ("NOT") on random noise
        if step % 20 == 0:
            noise = (np.random.rand(784) > 0.85).astype(np.float64)
            ai.train(noise, 10)

print("Training complete! Saving model...")
ai.save_model()