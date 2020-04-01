# MAT
The official implementation of the Molecule Attention Transformer. [ArXiv](https://arxiv.org/abs/2002.08264)

<p align='center'>
<img src="https://github.com/gmum/MAT/blob/master/assets/MAT.png" alt="architecture" width="600"/>
</p>

## Code
- `EXAMPLE.ipynb` jupyter notebook with an example of loading pretrained weights into MAT,
- `transformer.py` file with MAT class implementation,
- `utils.py` file with utils functions.

More functionality will be available soon!


## Pretrained weights
Pretrained weights are available [here](https://drive.google.com/open?id=11-TZj8tlnD7ykQGliO9bCrySJNBnYD2k)


## Results
In this section we present the average rank across the 7 datasets from our benchmark.

- Results for hyperparameter search budget of 500 combinations.
![](https://github.com/gmum/MAT/blob/master/assets/results_500.png)

- Results for hyperparameter search budget of 150 combinations.
![](https://github.com/gmum/MAT/blob/master/assets/results_150.png)

- Results for pretrained model
![](https://github.com/gmum/MAT/blob/master/assets/results_pretrained.png)


## Requirements
- PyTorch 1.4


## Acknowledgments
Transformer implementation is inspired by [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html).
