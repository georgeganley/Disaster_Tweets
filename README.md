# Disaster Tweets Prediction with BERT

This notebook originally comes from the Kaggle's "KerasNLP Starter Notebook Disaster Tweets". The original version uses Keras' DistilBERT pretrained model to identify tweets that are about or describe true disasters.

The notebook has been edited to use Hugging Face's transform library and pyTorch to solve the same problem.

The main goals of this project are:
- Import a pre-trained BERT model to classify the tweets.
- Evaluate model performance and explore areas for improvement.

## Dataset

The dataset used for this project consists of 10,000 tweets, each labeled as either a real disaster (1) or not (0). It includes the following columns:
- `id`: Unique identifier for each tweet.
- `keyword`: Keyword from the tweet (may be blank).
- `location`: Location from which the tweet was sent (may be blank).
- `text`: The text content of the tweet.
- `target`: The label (1 for real disaster, 0 for not).


### Data Preprocessing

- Split the data into training and validation sets.
- Tokenize the text data using `BertTokenizer` from the Hugging Face `transformers` library.
- Create PyTorch datasets for training, validation, and testing.

### Model Training

- Load a pre-trained BERT model (`bert-base-uncased`) using `BertForSequenceClassification`.
- Define training arguments and create a `Trainer` instance.
- Train the model on the training dataset.

## Outcome

Confusion matrices and key statistics are summarized below for the performance of a model created only with the pretrained BERT.

### In Sample Performance

- Accuracy for Training dataset: 0.922824302134647 
- Precision for  Training dataset: 0.9349232012934519 
- Recall for  Training dataset: 0.8821510297482837 
- F1 Score for  Training dataset: 0.9077708006279435

Confusion Matrix
![alt text](https://github.com/georgeganley/Disaster_Tweets/blob/main/Images/Training_confusion.png)

### Out Sample Performance

- Accuracy for Validation dataset: 0.8273145108338805 
- Precision for  Validation dataset: 0.8073248407643312 
- Recall for  Validation dataset: 0.7812018489984591 
- F1 Score for  Validation dataset: 0.7940485512920907

Confusion Matrix
![alt text](https://github.com/georgeganley/Disaster_Tweets/blob/main/Images/Validation_confusion.png)

## Takeaways and areas for improvement
With an F1 score on out-sample data just below 80%, it looks like BERT is doing a pretty good job of figuring out tweets that describe or are written from the context of a true disaster.
Still, there's room for improvement. One key observation is that the training and test data have a roughly 60/40 split of 'Not disaster' to 'Real disaster' tweet labels; in reality, one might not expect true disaster tweets to make up such a large portion of all disaster-esque tweets.

Keeping the data as-is, though, some fine tuning could improve performance, as the current results are based on an unadjusted, pre-trained BERT version.
