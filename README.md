# HeatFlow
## Interpreting Neural Networks Through the Lens ofHeat Flow

## Get Started

1. Install Python 3.7, JAX, Flax
2. Download MNIST and UTKFace dataset
3. Run the scripts, which includes 
    - Training the regression/classification model to be explained.
    - Training VAE model for manifold learning. (Skipped for Synthetic data).
    - Solving the heat equation model by deep methods.
    - Explaining by decomposing the Laplacian and comparing with other methods including Grad, Smooth Grad, IG and BlurIG.

    ```bash
    bash ./train/scripts/run_synthetic.sh
    bash ./train/scripts/run_mnist.sh
    ```


## Results

### UTKFace
Comparison of HeatFlow with vanilla Gradient(Grad), Smooth Gradient(SG), Integerated Gradients(IG) and Blur IG on facial age regression tasks.

![alt text](https://anonymous.4open.science/r/heat-explainer-FFD0/exp/face/face.png?raw=true)

### MNIST
Heat diffusion for MNIST samples comparing logits of different class. **Left**: Change in function value. **Right**: HeatFlow attribution maps.

![alt text](https://anonymous.4open.science/r/heat-explainer-FFD0/exp/mnist/mnist_compare.png?raw=true)