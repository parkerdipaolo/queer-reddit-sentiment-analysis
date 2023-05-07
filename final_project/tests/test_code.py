import pandas as pd

from sklearn.pipeline import Pipeline
from final_project.utils.data_files import concatenate
from final_project.utils.sentiment_analysis import extract_sentences, create_pipelines, run_pipelines


def test_concatenate():
    concatenate('test_data')
    test_df = pd.read_csv('test_data/Reddit_data.tsv', delimiter='\t')
    control_df = pd.DataFrame({'Title': ['Hello', 'Goodbye', 'Hello', 'Goodbye'],
                               'Body': ['world', 'BERT', 'BERT', 'world'],
                               'Comments': ['!', '.', '!', '.'],
                               'Post ID': [1, 2, 3, 4],
                               'Subreddit': ['a', 'b', 'c', 'd'],
                               'Score': [10, 10, 10, 10],
                               'Upvote ratio': [1, 1, 1, 1],
                               'Date': [0, 0, 0, 0]
                               })
    print(test_df)
    print(control_df)
    assert test_df.equals(control_df)


def test_extract_sentences():
    # create a test dataframe
    data = pd.DataFrame({'col1': ['hello', 'world'], 'col2': ['goodbye', 'bert']})
    expected_output = [{'sentence1': 'hello', 'sentence2': 'goodbye'}, {'sentence1': 'world', 'sentence2': 'bert'}]
    assert extract_sentences(data) == expected_output


def test_create_pipelines():
    # test pipeline creation
    id2label = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
    ots_pipeline, ft_pipeline = create_pipelines(id2label, label2id)
    assert isinstance(ots_pipeline, Pipeline)
    assert isinstance(ft_pipeline, Pipeline)


def test_run_pipelines():
    # test pipeline functionality
    id2label = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
    ots_pipeline, ft_pipeline = create_pipelines(id2label, label2id)
    rows = [{'sentence1': 'She is very strong', 'sentence2': 'Zie is very strong'}]
    ft_sentiment_label_array, ots_sentiment_label_array, ft_results, ots_results = run_pipelines(ots_pipeline,
                                                                                                 ft_pipeline, rows)
    assert ft_sentiment_label_array.shape == (1, 2)
    assert ots_sentiment_label_array.shape == (1, 2)
