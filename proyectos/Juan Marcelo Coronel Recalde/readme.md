# Sentiment Analysis with Fuzzy Logic System

This project performs sentiment analysis on text data using a combination of natural language processing (NLP) and fuzzy logic techniques. The system reads a dataset of sentences, preprocesses each sentence, calculates sentiment scores, and then classifies each sentence as positive, neutral, or negative using fuzzy logic.

## Features

- Text preprocessing: Removing URLs, punctuation, and stopwords, and applying lemmatization.
- Sentiment scoring with `sentiwordnet` and fuzzy logic-based classification.
- Fuzzy logic system to infer sentiment based on positive and negative scores.
- Outputs benchmark results, including sentiment classification and execution time.

## Dependencies

The project requires the following Python packages:

- `nltk`
- `scikit-fuzzy`
- `numpy`
- `pandas`
- `tqdm`

These dependencies are listed in the `requirements.txt` file. To install them, follow the setup instructions below.

## Setup Instructions

1. **Clone the repository** (or download the project files to your local system).

2. **Navigate to the project directory** in your terminal:

   ```bash
   cd /path/to/your/project
   ```

3. **Install the required packages** by running:
   ```bash
   pip install -r requirements.txt
   ```
   or
   ```bash
   pip3 install -r requirements.txt
   ```
4. **Decompress the files** from the archive.zip:
   Save the file train_data.csv and test_data.csv in the main folder "Juan Marcelo Coronel Recalde"

## Usage

Run the main scrip to begin the sentiment analysis:

```bash
python main.py or main_text.py (short version) in Windows.
```

```bash
python3 main.py or main_text.py (short version) in Unix based systems.
```
