# SingLoRA: A Minimal Implementation

This repository provides a minimal, single-file implementation of SingLoRA (Single Matrix Low-Rank Adaptation) as described in the paper ["SingLoRA: Low Rank Adaptation Using a Single Matrix"](https://arxiv.org/abs/2507.05566) by Bensaïd et al.

## Overview

SingLoRA is a parameter-efficient fine-tuning method that simplifies the LoRA architecture by using a single trainable matrix instead of two. This implementation demonstrates how to apply SingLoRA to transformer models using PyTorch and the Hugging Face Transformers library.

## Features

- Simple, self-contained implementation in a single Python file
- Compatible with Hugging Face Transformers models
- Includes a working example with DistilBERT
- Demonstrates parameter reduction compared to full fine-tuning

## Installation

```bash
pip install -r requirements.txt
```

## Usage

The implementation consists of two main components:

1. The SingLoRA implementation in `singlora/main.py`
2. A complete usage example in `example.py`

To run the example:

```bash
python example.py
```

The example demonstrates:
- Applying SingLoRA to a DistilBERT model
- Comparing model parameters before and after SingLoRA application
- Training the model on a dummy dataset
- Proper handling of the ramp-up period

## Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@misc{bensaïd2025singloralowrankadaptation,
      title={SingLoRA: Low Rank Adaptation Using a Single Matrix}, 
      author={David Bensaïd and Noam Rotstein and Roy Velich and Daniel Bensaïd and Ron Kimmel},
      year={2025},
      eprint={2507.05566},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2507.05566}, 
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
