import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix
from transformers import pipeline, BertForSequenceClassification, BertTokenizer


def extract_sentences(data):
    """
    Extracts the pairwise sentences from their respective columns in the data frame and puts them into a list of
    dictionaries
    :param data: dataframe containing formatted Winoqueer data
    :return: rows object containing sentence pairs
    """
    rows = []
    for index, row in data.iterrows():
        rows.append({
            'sentence1': row[0],
            'sentence2': row[1],
        })
    return rows


def create_pipelines(id2label, label2id):
    """
    Instantiates text classification pipelines for sentiment analysis using off-the-shelf ('bert-base-uncased') and
    fine-tuned ('bert-unsupervised') models
    :param id2label: mapping from integer ids to sentiment labels
    :param label2id: mapping from sentiment labels to integer ids
    :return: sentiment analysis pipelines from off-the-shelf and fine-tuned models
    """
    # Instantiate FT Pipeline
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    ots_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3, id2label=id2label,
                                                              label2id=label2id)
    ft_model = BertForSequenceClassification.from_pretrained('bert-unsupervised', num_labels=3, id2label=id2label,
                                                             label2id=label2id)

    ots_sentiment_pipeline = pipeline('text-classification', tokenizer=tokenizer, model=ots_model)
    ft_sentiment_pipeline = pipeline("text-classification", tokenizer=tokenizer, model=ft_model)

    return ots_sentiment_pipeline, ft_sentiment_pipeline


def run_pipelines(ots_sentiment_pipeline, ft_sentiment_pipeline, rows):
    """
    Runs the Winoqueer data through both pipelines.
    :param ots_sentiment_pipeline: sentiment analysis pipeline from off-the-shelf model
    :param ft_sentiment_pipeline: sentiment analysis pipeline from fine-tuned model
    :param rows: object containing extracted sentence pairs from winoqueer dataset
    :return: off-the-shelf results, off-the-shelf label vector; fine-tuned results, fine-tuned label vector
    """
    # Run the OTS pipeline on the data and create an array of sentiment labels
    ots_results = []
    ots_sentiment_label_vectors = []

    for row in rows:
        ots_result1 = ots_sentiment_pipeline(row['sentence1'])
        ots_result2 = ots_sentiment_pipeline(row['sentence2'])
        ots_sentiment_class1 = (ots_result1[0]['label'], ots_result1[0]['score'])
        ots_sentiment_class2 = (ots_result2[0]['label'], ots_result2[0]['score'])
        ots_results.append((row['sentence1'], ots_sentiment_class1, row['sentence2'], ots_sentiment_class2))
        ots_sentiment_label_vectors.append((ots_sentiment_class1[0], ots_sentiment_class2[0]))

    # create an array of sentiment labels
    ots_sentiment_label_array = np.array(ots_sentiment_label_vectors)
    print(ots_results)
    print(ots_sentiment_label_array)

    # Run the FT pipeline on the data and create an array of sentiment labels
    ft_results = []
    ft_sentiment_label_vectors = []

    for row in rows:
        ft_result1 = ft_sentiment_pipeline(row['sentence1'])
        ft_result2 = ft_sentiment_pipeline(row['sentence2'])
        ft_sentiment_class1 = (ft_result1[0]['label'], ft_result1[0]['score'])
        ft_sentiment_class2 = (ft_result2[0]['label'], ft_result2[0]['score'])
        ft_results.append((row['sentence1'], ft_sentiment_class1, row['sentence2'], ft_sentiment_class2))
        ft_sentiment_label_vectors.append((ft_sentiment_class1[0], ft_sentiment_class2[0]))

    # create an array of sentiment labels
    ft_sentiment_label_array = np.array(ft_sentiment_label_vectors)
    print(ft_results)
    print(ft_sentiment_label_array)
    return ft_sentiment_label_array, ots_sentiment_label_array, ft_results, ots_results


def label_index_array(label2id, ots_sentiment_label_array, ft_sentiment_label_array):
    """
    Converts the label arrays to their id values
    :param label2id: mapping from sentiment labels to integer ids
    :param ots_sentiment_label_array: label array for the off-the-shelf pipeline
    :param ft_sentiment_label_array: label array for the fine-tuned pipeline
    :return: id arrays based on corresponding label arrays
    """
    ots_sentiment_id_array = np.vectorize(label2id.get)(ots_sentiment_label_array)
    ft_sentiment_id_array = np.vectorize(label2id.get)(ft_sentiment_label_array)

    return ots_sentiment_id_array, ft_sentiment_id_array


def create_color_map():
    """
    Creates a custom, gradient color map based on our presentation color scheme
    :return: color map
    """
    # define custom colors using hex codes
    colors_nlp = ['#ebb55a', '#d84e2e', '#637b7f']
    positions = [0.0, 0.5, 1.0]

    # Create a LinearSegmentedColormap object
    nlp_cmap = LinearSegmentedColormap.from_list('my_colormap', list(zip(positions, colors_nlp)))

    return nlp_cmap


def plot_confusion_matrix(ft_sentiment_id_array, ots_sentiment_id_array, nlp_cmap):
    """
    Plot a confusion matrix with the OTS and FT sentiment labels
    :param ft_sentiment_id_array: id array for fine-tuned model
    :param ots_sentiment_id_array: id array for off-the-shelf model
    :param nlp_cmap: color map
    :return: confusion matrix showing label prediction overlap between fine-tuned and off-the-shelf models
    """
    # create the confusion matrix using ravel() because the arrays are multi-dimensional
    cm = confusion_matrix(ft_sentiment_id_array.ravel(), ots_sentiment_id_array.ravel())

    # Plot confusion matrix as heatmap
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, cmap=nlp_cmap, fmt='g', xticklabels=['NEG', 'NEU', 'POS'],
                yticklabels=['NEG', 'NEU', 'POS'])

    # Set labels and title
    ax.set_xlabel('Off-the-Shelf')
    ax.set_ylabel('Fine-Tuned')
    ax.set_title('Label Prediction Overlap')
    plt.show()
