# Vision Transformers

Implementation of [Vision Transformer](https://openreview.net/forum?id=YicbFdNTTy) in PyTorch, a new model to achieve SOTA in vision classification with using transformer style encoders. Associated [blog](https://abhaygupta.dev/blog/vision-transformer) article.

![ViT](./static/model.png)

## Features

- [x] Vanilla ViT
- [x] Hybrid ViT (with support for BiTResNets as backbone)
- [x] Hybrid ViT (with support for AxialResNets as backbone)
- [x] Training Scripts

To Do:

- [ ] Training Script
  - [ ] Support for linear decay
  - [ ] Correct hyper parameters
- [ ] Full Axial-ViT

## References

1. [BiTResNet](https://github.com/google-research/big_transfer/tree/master/bit_pytorch)
2. [AxialResNet](https://github.com/csrhddlam/axial-deeplab)

## Citations

```BibTeX
@inproceedings{
    anonymous2021an,
    title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
    author={Anonymous},
    booktitle={Submitted to International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=YicbFdNTTy},
    note={under review}
}
```
