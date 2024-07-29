# Disaster Tweets Prediction with BERT

This notebook originally comes from the Kaggle's "KerasNLP Starter Notebook Disaster Tweets". The original version uses Keras' DistilBERT pretrained model to identify tweets that are about or describe true disasters.

The original notebook has been edited to use Hugging Face's transform library and pyTorch to solve for the same issue.

The main goals of this project are:
- Load and preprocess a dataset of disaster-related tweets.
- Fine-tune a pre-trained BERT model to classify the tweets.
- Evaluate the model's performance using accuracy, precision, recall, and F1 score.
- Visualize the results using confusion matrices.

## Dataset

The dataset used for this project consists of 10,000 tweets, each labeled as either a real disaster (1) or not (0). It includes the following columns:
- `id`: Unique identifier for each tweet.
- `keyword`: Keyword from the tweet (may be blank).
- `location`: Location from which the tweet was sent (may be blank).
- `text`: The text content of the tweet.
- `target`: The label (1 for real disaster, 0 for not).


## Key Steps

### Data Loading and Exploration

- Load the dataset using `pandas`.
- Display basic information about the dataset such as shape, memory usage, and sample entries.
- Explore the text length statistics for both training and testing datasets.

### Data Preprocessing

- Split the data into training and validation sets.
- Tokenize the text data using `BertTokenizer` from the Hugging Face `transformers` library.
- Create PyTorch datasets for training, validation, and testing.

### Model Training

- Load a pre-trained BERT model (`bert-base-uncased`) using `BertForSequenceClassification`.
- Define training arguments and create a `Trainer` instance.
- Train the model on the training dataset.

### Evaluation

- Evaluate the model's performance on the training and validation datasets.
- Compute and display accuracy, precision, recall, and F1 score.
- Visualize confusion matrices for training and validation datasets.
