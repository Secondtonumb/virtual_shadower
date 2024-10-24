# Virtual Shadower
![Virtual Shadower](https://i.ibb.co/BnhYt7d/virtual-shadower-banner.jpg)

---

**Paper:**  
- [APSIPA ASC 2024](https://arxiv.org/abs/2410.02239)  
- [Audio Sample](https://secondtonumb.github.io/publication_demo/APSIPA_2024/index.html)

---

## Introduction

This project introduces a virtual shadowing system, inspired by seq2seq-vc and foreign accent conversion. Unlike traditional shadowing, where non-native speakers mimic native utterances, this system has native speakers shadow non-native speakers (L1-shadowing-L2). This approach serves as an intelligibility indicator for non-native speakers.

## Installation

### Editable installation with virtualenv   
```bash
git clone https://github.com/Secondtonumb/virtual_shadower
cd virtual_shadower/tools
make

## If make fails, try the following (compile by stages):
cd virtual_shadower/tools
make virtualenv.done
make pytorch.done
make seq2seq_vc.done
make s3prl-vc.done
make monotonic_align
make speechbertscore.done
```
---

### For reproduction of the APSIPA ASC 2024 paper:
```bash
cd egs/L1sL2/vs1
```
and follow the instructions in the README.md file.

## Acknowledgements

This repository draws inspiration and resources from the following projects:

- [seq2seq-vc](https://github.com/unilight/seq2seq-vc)  
- [s3prl-vc](https://github.com/unilight/s3prl-vc)  
- [ParallelWaveGan](https://github.com/kan-bayashi/ParallelWaveGAN/) (specifically, the [HuBERT_unit_vocoder_hifigan_style](https://github.com/kan-bayashi/ParallelWaveGAN/tree/master/egs/vctk/hubert_voc1))  
- [DiscreteSpeechMetrics](https://github.com/Takaaki-Saeki/DiscreteSpeechMetrics) (for SpeechBERTScore)

---

## Citation

```bibtex
@inprocessing{geng2024vs,
  title={A Pilot Study of Applying Sequence-to-Sequence Voice Conversion to Evaluate the Intelligibility of L2 Speech Using a Native Speaker's Shadowings},
  author={Geng, Haopeng and Saito, Daisuke and Nobuaki, Minematsu},
  journal={arXiv preprint arXiv:2410.02239},
  year={2024}
}

@inprocessing{geng2024simulating,
  title={Simulating Native Speaker Shadowing for Nonnative Speech Assessment with Latent Speech Representations},
  author={Geng, Haopeng and Saito, Daisuke and Nobuaki, Minematsu},
  booktitle={arXiv preprint arXiv:2409.11742},
  year={2024}
}