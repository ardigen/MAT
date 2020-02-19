# MAT
The official implementation of the Molecule Attention Transformer.

![](https://drive.google.com/uc?export=view&id=1KfaI-WmEdkSoHDQd4TdoajXDS9w77eTG)


## Code
- `load_weights.ipynb` jupyter notebook with an example of loading pretrained weights into MAT,
- `transformer.py` file with MAT class implementation,
- `utils.py` file with utils functions.

More functionality will be available soon!


## Pretrained weights
Pretrained weights are available [here](https://drive.google.com/open?id=11-TZj8tlnD7ykQGliO9bCrySJNBnYD2k)


## Results
In this section we present the average rank across the 7 datasets from our benchmark.

- Results for hyperparameter search budget of 500 combinations.
![](https://drive.google.com/uc?export=view&id=1H2qIg4cCvZuPrL-m3-RN0nlw942u6SvY)

- Results for hyperparameter search budget of 150 combinations.
![](https://drive.google.com/uc?export=view&id=1AnideQr3BFbqTDhZsxcnjmvuOvGRP_F3)

- Results for pretrained model
![](https://drive.google.com/uc?export=view&id=1JDmRLc_gl-HGTtsxLW1ASX35SGrqWkXm)


## Requirements
- PyTorch 1.4


## Acknowledgments
Transformer implementation is inspired by [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html).
