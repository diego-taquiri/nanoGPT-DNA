# nanoGPT-DNA

The simplest, fastest repository for training a small GPT on DNA sequences. It is inspired by [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy, repurposed to explore the fascinating regulatory syntax of DNA. The goal here is to train a transformer model, much like nanoGPT, but not on Shakespeare or WebText—instead, on the human genome hg38. The model will learn the syntax and language of DNA, and help us uncover the intricate code underlying biological regulation. Work in progress!

# Early Results

![Figure 1](/figures/output.png)

**Figure 1:** Training Loss Curve During Pre-Training

Some early results from the pre-training phase of our DNA language model. This phase employs a batch size of 1 million tokens, a conservative learning rate, and gradient accumulation with Distributed Data Parallelization (DDP) across two NVIDIA RTX 4090 GPUs. The model is trained autoregressively on the entire human genome, hg38, leveraging tokenization at the nucleotide level.

- **Loss Decreasing**: The training loss demonstrates a stable and consistent decrease, signaling effective learning. With an initial warming phase for the loss, the model adjusts smoothly to the task.
  
- **Improved Stability**: Compared to earlier experimental rounds, this training setup is more robust, aided by conservative hyperparameter tuning and greater batch size.

- **Peaks in Loss**: During training on repetitive sequences of the genome, the loss occasionally drops to zero. This indicates potential overfitting or memorization of these repetitive patterns, a phenomenon worth investigating in future iterations.

This result aligns with similar efforts in DNA language modeling, such as the Nucleotide Transformer, and provides a promising foundation for further evaluations. Future assessments will involve benchmarks like DART-Eval.

## Why nanoGPT-DNA?

We are building nanoGPT-DNA to explore foundational language modeling for genomics. The intent is simple: take a small GPT-like architecture (autoregressive, multi-layer transformer with attention) and train it to learn the regulatory language of DNA. Our tokenization is at the nucleotide level (A, T, C, G), and we train using a next-nucleotide prediction task, similarly to how GPT models are trained for natural language processing.

Why not masked token prediction like many DNA models? We're interested in autoregressive training here, to build intuition and probe whether such a straightforward approach can compete in understanding regulatory motifs and syntax.

## The Dataset: Human Genome hg38

The dataset used is hg38, a recent version of the human genome comprising roughly 3.2 billion nucleotides. To keep things simple, we're only training on one human genome for now—no other human genomes, no primates, and no other species. Just a single genome, with the aim to train quickly and iterate effectively.

The focus for now is on quick iteration and experimentation, training a small model in a reasonable amount of time—not trying to take on large foundational models like Nucleotide Transformer, [HyenaDNA](https://github.com/HazyResearch/hyena-dna), or DNA-BERT2. Those are great for large-scale results, but here we're aiming for simplicity, hackability, and a core implementation anyone can run and understand.

## Training the Model

The architecture follows a GPT-like design: multiple layers, causal attention, trained to predict the next nucleotide. To keep it lightweight, we are sticking to a small number of transformer layers, with a context window spanning around 1,000 to 5,000 nucleotides. The context window is key—it lets the model "see" the sequence around each nucleotide to predict what's next, and ideally capture complex regulatory relationships.

The training loop and model structure are heavily based on [nanoGPT](https://github.com/karpathy/nanoGPT), but adapted to work with the unique structure of DNA data. Training is fast, the code is minimal and easy to hack, and we're starting simple: a few transformer layers, just enough compute, and keeping the dataset lean (i.e., just one genome).

## Evaluation

The key evaluation benchmark will be [DART-Eval](https://github.com/kundajelab/dart-eval) from Anshul Kundaje, focusing on Task 1—the classification of regulatory vs. non-regulatory sequences using the last-layer embeddings. Many large DNA language models perform very well here, so the goal is for nanoGPT-DNA to at least outperform a bigram character-level language model, a simple predecessor to modern language models, using its context window effectively.

For context, a bigram model uses only the most recent nucleotide to predict the next one, while nanoGPT-DNA will use the whole 1,000–5,000 nucleotide window. We hope this added context gives it the edge over simple models, even if we're not competing with the best in class just yet.

## Tokenization

We're keeping tokenization extremely simple—it's just at the nucleotide level (A, T, C, G). No byte-pair encoding, no k-mer style segmentation. It's fundamental, but this level of granularity is perfectly suited for DNA data.

## Installation

```bash
pip install torch numpy
```

### Dependencies:

- PyTorch
- numpy

### Hardware

The GPU that we are going to use to train these models, for now, is as follows:

We are lucky to have two NVIDIA RTX 4090 GPUs in our lab, the [Mirko Zimic Lab](https://scholar.google.com/citations?hl=en&user=J7KkjscAAAAJ&view_op=list_works&sortby=pubdate). These GPUs provide us with ample power for our initial training runs and allow us to iterate quickly. Additionally, we have access to a Tesla T4 GPU through my personal AWS account in the cloud. While the T4 is not as powerful as the RTX 4090s, it serves as a backup—though, due to the hourly cost and lower performance, it will be used sparingly, likely only for final evaluations or small-scale experiments.

It is very challenging to obtain access to A100 GPUs in AWS. I barely even managed to secure a Tesla T4 instance, let alone something more powerful like an 8xA100 node. The demand for GPUs is incredible, and AI even won two Nobel Prizes this year—it's a crazy time for the field! For now, we'll stick to the RTX 4090s, as they are already set up and offer cost-effective and readily available power for our purposes.

### Future Directions

- Experimenting with training larger models, adding more transformer layers, and increasing the model size iteratively.
- Expanding training data to include other genomes, or even cross-species genomic data, to improve the model's understanding of broader genomic contexts.
- Introducing more advanced evaluation tasks, beyond Task 1 of [DART-Eval](https://github.com/kundajelab/dart-eval), to probe deeper regulatory understanding.
- Exploring finetuning on smaller-scale datasets of specific genomic regions or features.

### Acknowledgements

Big shout-out to [Andrej Karpathy](https://github.com/karpathy/nanoGPT) for the inspiration with nanoGPT, to Anshul Kundaje for the [DART-Eval](https://github.com/kundajelab/dart-eval) benchmark, and to the team behind [HyenaDNA](https://github.com/HazyResearch/hyena-dna) LLM, whose implementation we followed closely, although we used a vanilla transformer instead of SSM. This project draws heavily from nanoGPT, taking inspiration from the simple yet effective training approach.

All experiments are made possible thanks to open-source libraries and tools from the Python ecosystem. Special thanks to the computational biology community for laying the groundwork for foundational DNA language models!

Stay tuned for updates as we iterate on the model, run training experiments, and see how well a tiny GPT can learn the language of our DNA. Let's see if we can build something that’s not quite a "DNA Shakespeare," but at least able to tell us a little more about the code of life.
