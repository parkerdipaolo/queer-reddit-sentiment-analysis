import pandas as pd
import csv
import praw
import nltk

from datetime import datetime
from pathlib import Path
from nltk.tokenize import sent_tokenize
from emoji import demojize


def scrape_data(subreddit, query, sort='top', time_filter='all'):
    """
    Function to scrape Reddit according to a specified subreddit and keyword query using PRAW
    :param subreddit: name of the subreddit to be scraped
    :param query: keyword to query for posts in subreddit
    :param sort: sort mechanism for posts; can be one of "relevance", "hot", "top", "new", or "comments"
    (defaults to "top" in this package)
    :param time_filter: timeframe in which to search; can be one of "all", "day", "hour", "month", "week", or "year"
    (defaults to "all" in this package)
    :return: .tsv file with appropriate Reddit data and metadata from query
    """
    # create a reddit instance using appropriate API credentials
    reddit = praw.Reddit(client_id='AuBFnC_i2I91utJ5v_VpOA',
                         client_secret='cJ1chbl1541SbDPxQPQWCYZ7omPOnA',
                         user_agent='Scraping_Test',
                         username='pd702',
                         password='Gt4+k4L9U%M*Gz/')
    posts = reddit.subreddit(subreddit).search(query, sort=sort, time_filter=time_filter, limit=1000)
    # create a data file if one for the specified subreddit(s) and query combination does not already exist
    try:
        with open(f'../data/{subreddit}_{query}_data.tsv', 'x', newline='') as fp:
            writer = csv.writer(fp, delimiter='\t')
            writer.writerow(['Title', 'Body', 'Comments', 'Post ID', 'Subreddit', 'Score', 'Upvote ratio', 'Date'])
            count = 0
            for post in posts:
                comments = []
                submission = reddit.submission(post.id)
                submission.comments.replace_more(limit=None)
                for comment in submission.comments.list():
                    comments.append(comment)
                post_comments = {post.id: comments}
                comments_str = '\n'.join([comment.body.replace('\t', ' ') for comment in post_comments[post.id]])
                writer.writerow([post.title, post.selftext, comments_str,
                                post.id, post.subreddit, post.score, post.upvote_ratio,
                                 datetime.utcfromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S')])
                count += 1
                if count % 10 == 0:
                    print(f'{count} posts pulled.')
            print(f'{count} total posts pulled.')
        return fp
    # exception handling for exclusive creation if the data file already exists to avoid overwriting
    except FileExistsError:
        print('Data for this subreddit and query has already been pulled')


def concatenate(folder):
    """
    Function to concatenate several Reddit data/metadata files into a single .tsv file with duplicates removed
    :param folder: data directory containing individual .tsv files of Reddit data
    :return: single .tsv file containing Reddit data/metadata from all .tsv data files in the specified directory
    """
    # declare path variable and initialize empty dataframe
    path = Path(folder)
    master_df = pd.DataFrame()
    # iterate through each .tsv file in directory to create a dataframe and concatenate it to the main dataframe
    count = 0
    for f in path.glob('./*.tsv'):
        df = pd.read_csv(f, delimiter='\t')
        master_df = pd.concat([master_df, df])
        count += 1
        if count % 10 == 0:
            print(f'{count} files merged.')
    print(f'{count} files merged.')
    # once all data files have been added, drop duplicate posts and reset the index
    master_df.drop_duplicates(subset=['Title', 'Body', 'Post ID'], inplace=True)
    print('Duplicates dropped.')
    master_df.reset_index(drop=True, inplace=True)
    # save the dataframe to a .tsv file in the specified data directory
    master_df.to_csv(f'{folder}/Reddit_data.tsv', sep='\t', index=False)


def tokenize(text):
    """
    Function to tokenize a text input
    :param text: string to be tokenized
    :return: list of sentence-tokenized, demojized text
    """
    # split text by newline character (using as an additional sentence boundary in lieu of sentence-final punctuation)
    print('Splitting text into sentences.')
    sents = text.split('\n')
    # tokenize into sentences using nltk
    sents = [sent_tokenize(sent) for sent in sents]
    # remove empty sublists/sentences
    print('Removing empty sentences.')
    sents = [sent for sent in sents if sent]
    # demojize each sentence and strip of surrounding whitespace
    print('Demojizing sentences.')
    sents = [[demojize(sentence.strip()) for sentence in sent] for sent in sents]
    # flatten sentence sublists into a single list using list comprehension
    sents = [sentence for sent in sents for sentence in sent]
    return sents


def preprocess_tsv(tsv):
    """
    Function to preprocess concatenated .tsv data file by dropping unnecessary metadata and tokenizing text
    :param tsv: concatenated .tsv file with Reddit data and metadata
    :return: dataframe containing preprocessed Reddit data
    """
    # some comment about this once I figure out where we want it
    nltk.download('punkt')
    # create dataframe from .tsv file and drop columns with unnecessary metadata
    df = pd.read_csv(tsv, delimiter='\t')
    df = df.drop(['Post ID', 'Score', 'Upvote ratio', 'Date'], axis=1)
    # apply the tokenize function to the dataframe
    df = df.applymap(tokenize)
    return df


def benchmark_data(tsv):
    """
    Function to extract necessary data from the WinoQueer benchmark dataset and write it to a new file
    :param tsv: path to the WinoQueer .tsv file
    :return: new .tsv file where each row contains corresponding sentence pairs in two separate columns
    """
    path = Path(tsv)
    # read Winoqueer .tsv file
    with open(tsv, 'r', newline='') as wp:
        reader = csv.reader(wp, delimiter='\t')
        next(reader)
        # create new _pairs.tsv file in exclusive creation mode to avoid overwriting original file
        try:
            with open(f'../{path.stem}_pairs.tsv', 'x', newline='') as np:
                writer = csv.writer(np, delimiter='\t')
                # take each sentence pair and write constituent sentences to separate columns in the new file
                count = 0
                for row in reader:
                    row = str(row).split(',')
                    writer.writerow([row[1].strip(), row[2].strip()])
                    count += 1
                    if count % 1000 == 0:
                        print(f'{count} sentence pairs written.')
                print(f'{count} sentence pairs written.')
        # exception handling for exclusive creation if the data file already exists to avoid overwriting
        except FileExistsError:
            print('Split benchmark data file already exists')
