# CCAN

**Context-Aware Cross-Attention for Non-Autoregressive Translation** (COLING 2020).

We address a localness perception problem in NAT cross-attention: the decoder underuses source-side local context. CCAN combines global cross-attention with a local window around the aligned source position and a gating mechanism: `g·ATT(ψ,V) + (1−g)·ATT(L(ψ),V)` with `g = σ(WQ)`. Default: window size 9, applied to all 6 decoder layers. Implemented on CMLM (Mask-Predict).

[Paper](https://aclanthology.org/2020.coling-main.389/)

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

## Data

BPE 32K, sequence-level KD (paper uses distilled data). Example WMT14 En–De:

```bash
bash get_data.sh   # downloads and prepares data-bin/wmt14.en-de, etc.
```

Or set `DATA_DIR` and run `preprocess.py` with `--trainpref`, `--validpref`, `--testpref`, `--destdir`, `--joined-dictionary`, `--nwordssrc` / `--nwordstgt` (e.g. 32768).

## Training

6-layer encoder/decoder, d_model 512, 8 heads, FFN 2048 (paper §4.1). CCAN: `--decoder-layers-to-apply-local '1,2,3,4,5,6'` and `--win-size 9`.

```bash
export DATA_DIR=databin/ende/distill_de SAVE_DIR=wmt14ende_model
bash wmt14ende_pipeline.sh ccan_win9
```

Optional: `GPU=0,1,2,3`. Checkpoints: `SAVE_DIR/<run_name>/checkpoint_best.pt`. Paper averages top-3 by validation BLEU.

## Decode & eval

```bash
DATA=databin/ende/distill_de CHECKPOINT=wmt14ende_model/ccan_win9 SUBSET=test bash wmt14ende_eval.sh ccan_win9 0
```

With reference file: `REF=/path/to/test.de bash wmt14ende_eval.sh ...` to get sacrebleu. Output: `wmt14ende_rst/<run_name>/out.out` (or set `OUT`).

## Citation

```bibtex
@inproceedings{ding2020context,
  title={Context-Aware Cross-Attention for Non-Autoregressive Translation},
  author={Ding, Liang and Wang, Longyue and Wu, Di and Tao, Dacheng and Tu, Zhaopeng},
  booktitle={COLING},
  year={2020}
}
```
