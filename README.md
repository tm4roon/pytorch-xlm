# XLM: Cross-lingual Language Model Pretraining 
An implementation of [Cross-lingual Language Model Pretraining (XLM)](https://arxiv.org/abs/1901.07291) using pytorch.
You can choose following three training models. 
  - Causal language model ( `-—task causal`)
  - Masked language model ( `-—task masked`)
  - Translation language model ( `-—task translation`)

<p align=center>
<img src=https://user-images.githubusercontent.com/53220859/65595686-3fe9f680-dfd0-11e9-900a-e17128e17153.png>
</p>


## Settings
This code are depend on the following.
- python==3.6.5
- pytorch==1.1.0
- torchtext==0.3.1

```sh
git clone https://github.com/t080/pytorch-xlm.git
cd ./pytorch-xlm
pip install -r requirements.txt
```
<br>


## Usages
When a causal language model or a masked language model are trained, you must give a monolingual corpus (.txt) to the `--train` option.

```sh
python train.py \
  --task causal (or masked) \
  --train /path/to/train.txt \
  --savedir ./checkpoints \
  --gpu
```
<br>


When a translation language model is trained, you must give a parallel corpus (.tsv) to the `--train` option.

```sh
python train.py \
  --task translation \
  --train /path/to/train.tsv \
  --savedir ./checkpoints \
  --gpu
```
<br>


## References
- [Lample, Guillaume, and Alexis Conneau. "Cross-lingual language model pretraining." arXiv preprint arXiv:1901.07291 (2019).](https://arxiv.org/abs/1901.07291)
