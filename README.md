# Heat-Explainer

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