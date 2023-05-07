import csv
import praw

from datetime import datetime


def scrape_data(subreddit, query, sort='top', time_filter='all'):
    reddit = praw.Reddit(client_id='AuBFnC_i2I91utJ5v_VpOA',
                         client_secret='cJ1chbl1541SbDPxQPQWCYZ7omPOnA',
                         user_agent='Scraping_Test',
                         username='pd702',
                         password='Gt4+k4L9U%M*Gz/')
    posts = reddit.subreddit(subreddit).search(query, sort=sort, time_filter=time_filter, limit=1000)
    # search returns maximum of 1000 results; may need to run multiple times
    try:
        with open(f'../data/{subreddit}_{query}_data.tsv', 'x', newline='') as fp:
            writer = csv.writer(fp, delimiter='\t')
            writer.writerow(['Title', 'Body', 'Comments', 'Post ID', 'Subreddit', 'Score', 'Upvote ratio', 'Date'])
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
        return fp
    except FileExistsError:
        print('Data for this subreddit and query has already been pulled')


def quality_check(tsv, column):
    csv.field_size_limit(100000000)  # set the field size limit to 100 MB
    with open(tsv, 'r', newline='') as fp:
        reader = csv.reader(fp, delimiter='\t')
        for row in reader:
            print(len(row[column]))

    path = Path('../data')
    csv.field_size_limit(100000000)  # set the field size limit to 100 MB
    for f in path.glob('./*.tsv'):
        with open(f, 'r', newline='', encoding='utf-8') as fp:
            reader = csv.reader(fp, delimiter='\t')
            count = 0
            for row in reader:
                count += 1
            print(f'{f.name}: {count - 1} rows')