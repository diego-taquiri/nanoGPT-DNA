# nanoGPT-DNA

Training a small GPT on DNA sequences. It is inspired by [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy, repurposed to explore the  regulatory syntax of DNA. The goal here is to train a transformer model, much like nanoGPT, but not on Shakespeare or WebText—instead, on the human genome hg38. The model will learn the syntax and language of DNA, and help us uncover the code underlying biological regulation. Work in progress!

# Early Results
![Figure 1](/figures/1st-good-run.png)

**Figure 1:**  Training/Validation Loss Curve During Pre-Training on 85M-Parameter Model Trained Over 13 Billion Tokens

![Figure 2](/figures/dart-eval-results1.png)

**Figure 2:** DART-Eval Accuracy Comparison for Regulatory vs. Shuffled Control Sequences

The model is pre-trained using a self-supervised task of Next Token Prediction, where it learns to predict the next nucleotide in a sequence. This task enables the model to capture the statistical dependencies, syntax, and patterns of DNA. Some early results from the pre-training phase of our 85M params DNA language model. This phase employs a batch size of 1 million tokens per step, a less conservative learning rate, and gradient accumulation with Distributed Data Parallelization (DDP) across two NVIDIA RTX 4090 GPUs. The model is trained autoregressively on the human genome, hg38, leveraging tokenization at the nucleotide level.

- **Loss Decreasing**: The training loss demonstrates a stable and consistent decrease, signaling effective learning. With an initial warming phase for the loss, the model adjusts smoothly to the task.
  
- **Improved Stability**: Compared to earlier experimental rounds, this training setup is more robust, aided by hyperparameter tuning and greater batch size.

- **Optimized Data Loading**: Perhaps the most significant improvement came from refactoring the DataLoader to train on curated BED-defined genomic regions. By excluding gaps and unmappable regions, we achieved better training stability and memory efficiency.


### DART-Eval Benchmark Results

The DART-Eval benchmark evaluates the ability of DNA language models to differentiate between regulatory sequences and shuffled control sequences. This task leverages the model’s understanding of DNA syntax and statistical patterns to assign higher likelihoods to true regulatory sequences. By training on the entire human genome, hg38, we expect the model to learn both regulatory and non-regulatory sequences. Given that only 2% of the genome consists of coding regions, this pre-training equips the model to differentiate between regulatory elements and non-regulatory regions.

Key observations:

- **Benchmark Performance**: Our model achieved an accuracy of **0.64**, demonstrating non-random performance (random baseline = 0.5). While not yet state-of-the-art (e.g., Nucleotide Transformer: 0.745), these results are promising for a first evaluation.

- **Model Understanding**: This performance reflects the model’s capacity to learn the dependencies and patterns in the human genome, including differentiating regulatory from non-regulatory sequences.

- **Comparison**: Figure 2 highlights the accuracy of our model against several DNA language models, showing competitive early results despite limited training time and model scale. Further tuning and scaling could improve these outcomes.

---

### Future Directions

Several strategies will be explored to enhance model performance:

- **Extended Training**: Increasing training time to allow for further refinement of learned patterns.
- **Scaling Parameters**: Testing larger model sizes to capture more complex genomic structures.
- **Hyperparameter Optimization**: Systematic tuning of the learning rate, batch size, and other hyperparameters.
- **Randomized Epochs**: Introducing more variability in training data to prevent overfitting.
- **Advanced Benchmarks**: Expanding evaluations to include additional datasets and benchmarks.

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

### Long-term Directions

- Experimenting with training larger models, adding more transformer layers, and increasing the model size iteratively.
- Expanding training data to include other genomes, or even cross-species genomic data, to improve the model's understanding of broader genomic contexts.
- Exploring finetuning on smaller-scale datasets of specific genomic regions or features.

### Acknowledgements

Big shout-out to [Andrej Karpathy](https://github.com/karpathy/nanoGPT) for the inspiration with nanoGPT, to Anshul Kundaje for the [DART-Eval](https://github.com/kundajelab/dart-eval) benchmark, and to the team behind [HyenaDNA](https://github.com/HazyResearch/hyena-dna) LLM, whose implementation we followed closely, although we used a vanilla transformer instead of SSM. This project draws heavily from nanoGPT, taking inspiration from the simple yet effective training approach.

All experiments are made possible thanks to open-source libraries and tools from the Python ecosystem. Special thanks to the computational biology community for laying the groundwork for foundational DNA language models!

Stay tuned for updates as we iterate on the model, run training experiments, and see how well a tiny GPT can learn the language of our DNA. Let's see if we can build something that’s not quite a "DNA Shakespeare," but at least able to tell us a little more about the code of life.
