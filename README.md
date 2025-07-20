# Abstractive Text Summarization with LSTM and Attention

This project implements an **abstractive text summarization** model using a custom-built **sequence-to-sequence (Seq2Seq)** architecture with **LSTM layers** and **Bahdanau-style attention**. The model is trained from scratch on two datasets to generate human-readable summaries from longer text inputs.

---

## 🧠 Project Overview

Abstractive summarization aims to generate concise summaries that may include words or phrases not present in the original text. Unlike extractive summarization, which selects exact sentences, this approach requires the model to learn how to paraphrase content.

We implemented an encoder-decoder framework with attention to better capture semantic meaning and handle long sequences.

---

## 📚 Datasets

We used two datasets obtained from [Kaggle](https://www.kaggle.com/):

1. **Amazon Fine Food Reviews**
   - Fields used: `Text`, `Summary`
   - 100,000 examples
2. **India News Summary**
   - Fields used: `Text`, `Headlines`
   - 50,000 examples

Each dataset underwent preprocessing steps including:
- Lowercasing
- HTML tag removal
- Stopword removal
- Punctuation and special character cleanup
- Tokenization and vocabulary generation

---

## 🏗️ Model Architecture

- **Encoder**: Stacked LSTM layers that encode the input sequence into a fixed-length context vector
- **Decoder**: LSTM layers with **Bahdanau attention mechanism**
- **Decoding**: Greedy search (selecting the most likely word at each time step)
- **Loss Function**: Sparse categorical crossentropy
- **Training**: 90/10 train/validation split, 50 epochs, early stopping on validation loss

---

## 📊 Evaluation

- **Metric**: BLEU (Bilingual Evaluation Understudy) score
- **Observation**:
  - BLEU-1 scores were generally low for very short summaries
  - India News summaries performed better due to longer length and richer vocabulary
  - Human evaluation is still needed in cases where BLEU fails to reflect semantic accuracy

---

## 🧪 Sample Results

| Dataset        | BLEU Score Range | Summary Quality Notes                         |
|----------------|------------------|-----------------------------------------------|
| Amazon Reviews | 0.0 – 0.1        | Short summaries → low BLEU despite relevance  |
| India News     | 0.1 – 0.4        | Longer summaries → more overlap, better score |

---

## ▶️ How to Run

### 1. Install dependencies

```bash
pip install tensorflow pandas numpy matplotlib sklearn
