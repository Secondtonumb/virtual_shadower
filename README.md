# Virtual Shadower
![Virtual Shadower](virtual_shadower_banner.jpg)
---
+ Paper: [APSIPA ASC 2024](https://arxiv.org/abs/2410.02239), [AudioSample](https://secondtonumb.github.io/publication_demo/APSIPA_2024/index.html)
---
## Introduction
We follow the concept of seq2seq-vc and foreign accent conversion to make a virtual shadowing system. Different from typical shadowing where the non-native speaker shadows the native utterances, we propose a virtual shadowing system where the native speaker shadows the non-native speaker (L1-shadowing-L2), serving as an intelligibility indicator for the non-native speaker.
---
## Acknowledgements
This repo is mostly inspired by and heritaged from following repos:
+ [seq2seq-vc](https://github.com/unilight/seq2seq-vc) 
+ [s3prl-vc](https://github.com/unilight/s3prl-vc)
+ [ParallelWaveGan](https://github.com/kan-bayashi/ParallelWaveGAN/) especially the [HuBERT_unit_vocoder_hifigan_style](https://github.com/kan-bayashi/ParallelWaveGAN/tree/master/egs/vctk/hubert_voc1)
+ [DiscreteSpeechMetrics](https://github.com/Takaaki-Saeki/DiscreteSpeechMetrics) For SpeechBERTScore
---

## Citation
```
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

```