# XLM: Cross-lingual Language Model Pretraining 

[Cross-lingual Language Model Pretraining](https://arxiv.org/abs/1901.07291)のpytorch実装。以下の3種類の言語モデルを利用可能。

- Causal Language Model ( —task causal)
- Masked Language Model ( —task masked)
- Translation Language Model ( —task translation)
<br>



## 環境構築

```python3
git clone https://github.com/marucha80t/pytorch-xlm.git
cd ./pytorch-xlm
pip install -r requirements.txt
```

<br>



## 使用方法

Causal language model / Masked language modelを学習する場合は、単言語コーパスを渡す。

```python3
python train.py --task casual(or masked) \
                --train ./data/sample_train.txt \
                --valid ./data/sample_valid.txt \
                --savedir ./checkpoints \
                --gpu
```



Translation language modelを学習する場合は、並列コーパスを渡す。

```python3
python train.py --task casual(or masked) \
                --train ./data/sample_train.tsv \
                --valid ./data/sample_valid.tsv \
                --savedir ./checkpoints \
                --gpu
```

<br>



## 参考

- [Cross-lingual Language Model Pretraining](https://arxiv.org/abs/1901.07291)
