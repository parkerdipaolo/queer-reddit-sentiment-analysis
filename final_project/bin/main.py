import argparse
import pandas as pd

from final_project.utils.data_files import scrape_data, concatenate, preprocess_tsv, benchmark_data
from final_project.utils.fine_tuning import finetune_bert
from final_project.utils.sentiment_analysis import extract_sentences, create_pipelines, run_pipelines, \
    label_index_array, create_color_map, plot_confusion_matrix


def main():
    parser = argparse.ArgumentParser(description='Code for performing name matching')
    parser.add_argument('-task', help='task to perform: scrape Reddit for data, fine-tune BERT model, '
                                      'or conduct sentiment analysis', choices=['scrape', 'finetune', 'sentiment'],
                        required=True)
    parser.add_argument('-sr', help='subreddit(s) to scrape;'
                                    'if pulling from more than one at a time, concatenate using the following format:'
                                    'subreddit1+subreddit2')
    parser.add_argument('-q', help='query to search subreddit(s) for')
    parser.add_argument('-f', help='data directory')

    args = parser.parse_args()

    if args.task == 'scrape':
        # perform scraping according to specified subreddit(s) and query
        if args.sr and args.q:
            scrape_data(args.sr, args.q)
        # error handling if no subreddit or query is specified
        else:
            raise ValueError('Must specify a subreddit and query for scraping')

    if args.task == 'finetune':
        # error handling if no data directory is passed
        if not args.f:
            raise ValueError('Must specify a data directory containing individual data files')
        # concatenate individual data files, preprocess concatenated file, and fine-tune BERT on preprocessed data
        else:
            concatenate(args.f)
            df = preprocess_tsv('../Reddit_data.tsv')
            finetune_bert(df)

    if args.task == 'sentiment':
        # create id and label mappings
        id2label = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
        label2id = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}

        # parse Winoqueer data and extract sentence pairs
        benchmark_data('../winoqueer_benchmark.tsv')
        data = pd.read_csv('../winoqueer_benchmark_pairs.tsv', delimiter='\t', header=None)
        rows = extract_sentences(data)

        # create and run sentiment analysis classifiers for off-the-shelf and fine-tuned models
        ots_sentiment_pipeline, ft_sentiment_pipeline = create_pipelines(id2label, label2id)
        ft_sentiment_label_array, ots_sentiment_label_array, ft_results, ots_results =\
            run_pipelines(ots_sentiment_pipeline, ft_sentiment_pipeline, rows)
        ots_sentiment_id_array, ft_sentiment_id_array = label_index_array(label2id, ots_sentiment_label_array,
                                                                          ft_sentiment_label_array)

        # create confusion matrix to visualize results
        nlp_cmap = create_color_map()
        plot_confusion_matrix(ft_sentiment_id_array, ots_sentiment_id_array, nlp_cmap)
        print(plot_confusion_matrix)


if __name__ == '__main__':
    main()
