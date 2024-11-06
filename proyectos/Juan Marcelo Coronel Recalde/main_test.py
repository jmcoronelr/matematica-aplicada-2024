import nltk
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np
import pandas as pd
import os
import time
from nltk.corpus import sentiwordnet as swn
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords, wordnet
from tqdm import tqdm
import re
import string
tqdm.pandas()
nltk.download('sentiwordnet')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
nltk.download('punkt')

data_dir = "../Juan Marcelo Coronel Recalde/"
data = pd.read_csv(os.path.join(data_dir, 'test_data.csv'))

# Preproccesing text


def process_text(text: str):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower().encode('ascii', 'ignore').decode()
    if (text):
        return text
    else:
        return None


print('Number of samples in DataSet before cleaning: ', len(data))
print('Cleaning...')
data['sentence'] = data['sentence'].progress_map(process_text)
data = data.dropna(subset=['sentence']).reset_index(
    drop=True)  # Deletes rows with empty values

print('Number of samples in DataSet after cleaning: ', len(data))

cleaning_result = pd.DataFrame(data)
cleaning_result.to_csv('cleaned_dataset_test.csv', index=False)

data = pd.read_csv(os.path.join(data_dir, 'cleaned_dataset_test.csv'))

# Functions for sentiment process
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


# Preprocess the given text deleting stopwords, cleaning, tokenization and
def preprocess_text(text):
    text = text.translate(str.maketrans(
        '', '', string.punctuation))  # Deletes punctuation
    tokens = word_tokenize(text)  # Divides the text in tokens
    tokens = [word for word in tokens if word not in stop_words]
    pos_tags = pos_tag(tokens)

    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    processed_tokens = []
    for word, tag in pos_tags:
        wn_tag = get_wordnet_pos(tag)
        processed_tokens.append(lemmatizer.lemmatize(  # Lematizacion and creation of processed tokens
            word, wn_tag) if wn_tag else word)
    return processed_tokens


# Calculates the sentiment score
def get_sentiment_scores(words):
    pos_score = 0
    neg_score = 0
    for word in words:
        synsets = list(swn.senti_synsets(word))
        if not synsets:
            continue
        synset = synsets[0]
        pos_score += synset.pos_score()
        neg_score += synset.neg_score()
    total_words = len(words)

    # Normalization [0,1]
    return pos_score / total_words, neg_score / total_words


# Fuzzy logic system initialization
x_pos = np.arange(0, 1.1, 0.1)
x_neg = np.arange(0, 1.1, 0.1)
x_op = np.arange(0, 10.1, 0.1)

# Define membership functions and rules
pos_score_ctrl = ctrl.Antecedent(x_pos, 'positive_score')
neg_score_ctrl = ctrl.Antecedent(x_neg, 'negative_score')
output_sentiment = ctrl.Consequent(x_op, 'output_sentiment')

# Membership functions for inputs
min = 0
max = 1
mid = (min+max)/2
pos_score_ctrl['low'] = fuzz.trimf(x_pos, [min, min, mid])
pos_score_ctrl['medium'] = fuzz.trimf(x_pos, [min, mid, max])
pos_score_ctrl['high'] = fuzz.trimf(x_pos, [mid, max, max])

neg_score_ctrl['low'] = fuzz.trimf(x_neg, [min, min, mid])
neg_score_ctrl['medium'] = fuzz.trimf(x_neg, [min, mid, max])
neg_score_ctrl['high'] = fuzz.trimf(x_neg, [mid, max, max])

output_sentiment['negative'] = fuzz.trimf(x_op, [0, 0, 5])
output_sentiment['neutral'] = fuzz.trimf(x_op, [0, 5, 10])
output_sentiment['positive'] = fuzz.trimf(x_op, [5, 10, 10])

# Define rules and control system
rules = [
    ctrl.Rule(pos_score_ctrl['low'] &
              neg_score_ctrl['low'], output_sentiment['neutral']),
    ctrl.Rule(pos_score_ctrl['medium'] &
              neg_score_ctrl['low'], output_sentiment['positive']),
    ctrl.Rule(pos_score_ctrl['high'] &
              neg_score_ctrl['low'], output_sentiment['positive']),
    ctrl.Rule(pos_score_ctrl['low'] & neg_score_ctrl['medium'],
              output_sentiment['negative']),
    ctrl.Rule(pos_score_ctrl['medium'] &
              neg_score_ctrl['medium'], output_sentiment['neutral']),
    ctrl.Rule(pos_score_ctrl['high'] &
              neg_score_ctrl['medium'], output_sentiment['positive']),
    ctrl.Rule(pos_score_ctrl['low'] & neg_score_ctrl['high'],
              output_sentiment['negative']),
    ctrl.Rule(pos_score_ctrl['medium'] &
              neg_score_ctrl['high'], output_sentiment['negative']),
    ctrl.Rule(pos_score_ctrl['high'] &
              neg_score_ctrl['high'], output_sentiment['neutral']),
]
sentiment_ctrl = ctrl.ControlSystem(rules)
sentiment_simulation = ctrl.ControlSystemSimulation(sentiment_ctrl)

# Benchmarking loop
result_data = []
total_time = 0

# Counters for classification
classification_counts = {
    'positive': 0,
    'neutral': 0,
    'negative': 0
}

for index, row in tqdm(data.iterrows(), total=len(data)):
    start_time = time.time()
    preprocessed_text = preprocess_text(row['sentence'])
    if (preprocessed_text):
        pos_score, neg_score = get_sentiment_scores(preprocessed_text)

        sentiment_simulation.input['positive_score'] = pos_score
        sentiment_simulation.input['negative_score'] = neg_score
        sentiment_simulation.compute()
        result = sentiment_simulation.output['output_sentiment']

        if result > 6.7:
            label = 'positive'
        else:
            if result > 3.3:
                label = 'neutral'
            else:
                label = 'negative'
        exec_time = time.time() - start_time
        total_time += exec_time

        # Update count classification
        classification_counts[label] += 1

        result_data.append([
            row['sentence'],  # Original sentence
            row['sentiment'],  # Original label sentiment
            label,              # Sentiment
            pos_score,            # Positive score
            neg_score,            # Negative score
            result,               # Inference result
            exec_time             # Execution time
        ])

# DataFrame creation and output to csv
columns = [
    'original_sentence',
    'original_sentiment',
    'sentiment',
    'positive_score',
    'negative_score',
    'inference_result',
    'execution_time'
]
result_df = pd.DataFrame(result_data, columns=columns)
result_df.to_csv('sentiment_benchmark_test.csv', index=False)

# Calculate and show average time
avg_exec_time = total_time / len(data)
print("Tiempo de Ejecución Promedio Total:", avg_exec_time)

# Print classification
pos_count = classification_counts['positive']
neg_count = classification_counts['negative']
neu_count = classification_counts['neutral']
total = len(data)
print("Total: ", total)
print(f"Positivo: {pos_count}  {(pos_count/total)*100}%")
print(f"Neutral: {neu_count}  {(neu_count/total)*100}%")
print(f"Negativo: {neg_count}  {(neg_count/total)*100}%")
