# L1 shadowing L2
![](https://raw.githubusercontent.com/Secondtonumb/publication_demo/refs/heads/main/ICASSP_2025/figure/VS_model.png)
We follow the latent space conversion (LSC) method proposed by [Seq2seq/vc2/lsc](https://github.com/unilight/seq2seq-vc/tree/main/egs/l2-arctic/lsc), specifically.  
We adapted the AAS-VC conversion to guarantee the alignment performance.

## Data Preparation
Ask for the data from the author: [Email](mailto:kevingenghaopeng@gavo.u-tokyo.ac.jp)

## Preparation

```bash
## line 20 - 32 in run.sh:

db_root=/home/kevingenghaopeng/vc/seq2seq-vc/egs/arctic/vc2/downloads # database root directory
dumpdir=dump                # directory to dump full features
srcspk="V000_R_max_valid"   # V000_R_max_valid (L2R) or V006_SS_max_valid (L1SS)
trgspk="V006_SS_max_valid"  # V006_SS_max_valid (L1SS) or V006_S1_max_valid (L1S1)
num_train=2000              # number of training samples, set in local/data_prep_gavo.sh as well
stats_ext=h5
norm_name='self'            # used to specify normalized data.
                            # Ex: `judy` for normalization with pretrained model, `self` for self-normalization

src_feat=hubert             # source feature type, hubert, ppg_sxliu, or mel
trg_feat=ppg_sxliu          # target feature type, hubert, ppg_sxliu, or mel
dp_feat=hubert              # duration prediction feature type, hubert, ppg_sxliu, or mel
feat_layer=9                # for hubert only, specify the layer to extract features from, 9th layer is selected following paper's implementation if None, use the average of all layers.
```

## Usage:
```
## Download the pre-trained vocoder models
. ./run.sh --stage -1 --stop-stage 1

## Train/Dev/Test data preparation
. ./run.sh --stage 0 --stop-stage 0

## Feature extraction
. ./run.sh --stage 1 --stop-stage 1

## Normalization
. ./run.sh --stage 2 --stop-stage 2

## Training
. ./run.sh --stage 3 --stop-stage 3

## Decoding
. ./run.sh --stage 4 --stop-stage 4

## Evaluation
. ./run.sh --stage 5 --stop-stage 5
```