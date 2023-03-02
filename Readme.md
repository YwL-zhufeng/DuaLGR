# DuaLGR: Dual Label-Guided Graph Refinement for Multi-View Graph Clustering

This is the source code for paper: Dual Label-Guided Graph Refinement for Multi-View Graph Clustering, accepted at AAAI 2023.

# Requirements

- 'requirements.txt'
- The experiments are conducted on a Linux machine with a NVIDIA GeForce RTX 3070 GPU and Intel(R) Xeon(R)  E5-2678 v3 @ 2.50GHz CPU. 



# Datasets

ACM, ACM (HR 0.00) and ACM (HR 0.20) datasets are included in `./data/`, Texas, DBLP and Chameleon are publicly available, and other synthetic data will be published in the future.

## Raw data

|  Dataset  | #Clusters | #Nodes | #Features |                           Graphs                            |              HR              |
| :-------: | :-------: | :----: | :-------: | :---------------------------------------------------------: | :--------------------------: |
|    ACM    |     3     |  3025  |   1830    |            $\mathcal{G}^1$ <br />$\mathcal{G}^2$            |       0.82 <br />0.64        |
|   DBLP    |     4     |  4057  |    334    | $\mathcal{G}^1$ <br />$\mathcal{G}^2$ <br />$\mathcal{G}^3$ | 0.80 <br /> 0.67 <br /> 0.32 |
|   Texas   |     5     |  183   |   1703    |                       $\mathcal{G}^1$                       |             0.09             |
| Chameleon |     5     | 22777  |   2325    |                       $\mathcal{G}^1$                       |             0.23             |

## Synthetic data

|    Dataset    | #Clusters | #Nodes | #Features |                Graphs                 |       HR        |
| :-----------: | :-------: | :----: | :-------: | :-----------------------------------: | :-------------: |
| ACM (HR 0.00) |     3     |  3025  |   1830    | $\mathcal{G}^1$ <br />$\mathcal{G}^2$ | 0.00 <br />0.00 |
| ACM (HR 0.10) |     3     |  3025  |   1830    | $\mathcal{G}^1$ <br />$\mathcal{G}^2$ | 0.10 <br />0.10 |
| ACM (HR 0.20) |     3     |  3025  |   1830    | $\mathcal{G}^1$ <br />$\mathcal{G}^2$ | 0.20 <br />0.20 |
| ACM (HR 0.30) |     3     |  3025  |   1830    | $\mathcal{G}^1$ <br />$\mathcal{G}^2$ | 0.30 <br />0.30 |
| ACM (HR 0.40) |     3     |  3025  |   1830    | $\mathcal{G}^1$ <br />$\mathcal{G}^2$ | 0.40 <br />0.40 |
| ACM (HR 0.50) |     3     |  3025  |   1830    | $\mathcal{G}^1$ <br />$\mathcal{G}^2$ | 0.50 <br />0.50 |

# Test DuaLGR

```python
# Test DuaLGR on ACM dataset
python DuaLGR.py --dataset 'acm' --train False --use_cuda True --cuda_device 0

# Test DuaLGR on DBLP dataset
python DuaLGR.py --dataset 'dblp' --train False --use_cuda True --cuda_device 0

# Test DuaLGR on Texas dataset
python DuaLGR.py --dataset 'texas' --train False --use_cuda True --cuda_device 0

# Test DuaLGR on Chameleon dataset
python DuaLGR.py --dataset 'chameleon' --train False --use_cuda True --cuda_device 0

# Test DuaLGR on synthetic ACM dataset with HR 0.00
python DuaLGR.py --dataset 'acm00' --train False --use_cuda True --cuda_device 0

# Test DuaLGR on synthetic ACM dataset with HR 0.10
python DuaLGR.py --dataset 'acm01' --train False --use_cuda True --cuda_device 0

# Test DuaLGR on synthetic ACM dataset with HR 0.20
python DuaLGR.py --dataset 'acm02' --train False --use_cuda True --cuda_device 0

# Test DuaLGR on synthetic ACM dataset with HR 0.30
python DuaLGR.py --dataset 'acm03' --train False --use_cuda True --cuda_device 0

# Test DuaLGR on synthetic ACM dataset with HR 0.40
python DuaLGR.py --dataset 'acm04' --train False --use_cuda True --cuda_device 0

# Test DuaLGR on synthetic ACM dataset with HR 0.50
python DuaLGR.py --dataset 'acm05' --train False --use_cuda True --cuda_device 0
```

# Train DuaLGR

```python
# Train DuaLGR on ACM dataset
python DuaLGR.py --dataset 'acm' --train True --use_cuda True --cuda_device 0

# Train DuaLGR on DBLP dataset
python DuaLGR.py --dataset 'dblp' --train True --use_cuda True --cuda_device 0

# Train DuaLGR on Texas dataset
python DuaLGR.py --dataset 'texas' --train True --use_cuda True --cuda_device 0

# Train DuaLGR on Chameleon dataset
python DuaLGR.py --dataset 'chameleon' --train True --use_cuda True --cuda_device 0

# Train DuaLGR on synthetic ACM dataset with HR 0.00
python DuaLGR.py --dataset 'acm00' --train True --use_cuda True --cuda_device 0

# Train DuaLGR on synthetic ACM dataset with HR 0.10
python DuaLGR.py --dataset 'acm01' --train True --use_cuda True --cuda_device 0

# Train DuaLGR on synthetic ACM dataset with HR 0.20
python DuaLGR.py --dataset 'acm02' --train True --use_cuda True --cuda_device 0

# Train DuaLGR on synthetic ACM dataset with HR 0.30
python DuaLGR.py --dataset 'acm03' --train True --use_cuda True --cuda_device 0

# Train DuaLGR on synthetic ACM dataset with HR 0.40
python DuaLGR.py --dataset 'acm04' --train True --use_cuda True --cuda_device 0

# Train DuaLGR on synthetic ACM dataset with HR 0.50
python DuaLGR.py --dataset 'acm05' --train True --use_cuda True --cuda_device 0
```

**Parameters**: More parameters and descriptions can be found in the script and paper.

# Results of DuaLGR

|               | NMI% | ARI% | ACC% | F1%  |
| :-----------: | :--: | :--: | :--: | :--: |
|      ACM      | 73.2 | 79.4 | 92.7 | 92.7 |
|     DBLP      | 75.5 | 81.7 | 92.4 | 91.8 |
|     Texas     | 36.6 | 27.8 | 57.4 | 43.3 |
|   Chameleon   | 18.6 | 13.5 | 42.1 | 41.1 |
| ACM (HR 0.00) | 61.6 | 68.5 | 88.3 | 88.3 |
| ACM (HR 0.10) | 62.0 | 69.1 | 88.6 | 88.5 |
| ACM (HR 0.20) | 62.9 | 70.0 | 89.0 | 89.0 |
| ACM (HR 0.30) | 63.7 | 71.0 | 89.4 | 89.4 |
| ACM (HR 0.40) | 93.7 | 96.5 | 98.8 | 98.8 |
| ACM (HR 0.50) | 99.5 | 99.8 | 99.9 | 99.9 |

