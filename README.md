# Generate and Segment dental radiography: a comparative study of generative models

This repository aims to implement several generative techniques to both generate and segment dental images. It is organised as follow:


## `GAN` folder
Wasserstein GAN for generation. Modify the parameters in `main.py`, and run the training using `python main.py`.
You can change the model in `model.py`, and evaluate the model using `eval.py`.

## `baseline` folder

## `diffusion_scratch` folder


Finally, `split_data.py` is a script to split your dataset into train, validation and test sets that you can modify according to your data paths. 