# Evaluating Anti-Queer Bias in BERT
This project scraped Reddit subreddits (/trans and /NonBinary) to fine-tune BERT using domain-specific language. The off-the-shelf and fine-tuned models were then used in a sentiment analysis task on [Katy Felkner's Winoqueer dataset](https://github.com/katyfelkner/winoqueer) to compare sentiment label predictions.

Project by Parker DiPaolo, Sophie Henry, Kris Cook, and Sydney Toltz.

## Installation
To install the necessary packages and dependencies, navigate to the root directory of the project and create a virtual environment with conda:

```
conda activate
conda env create -f environment.yml
conda activate final_project
```

Install the project:

```
pip install .
```

## Design

This project is divided into three separate tasks:

1. Data scraping from Reddit
```
(task) scrape -sr <subreddit> -q <keyword query>
```
2. Fine-tuning BERT
```
(task) finetune -f <data directory>
```
3. Conducting sentiment analysis on Winoqueer data
```
(task) sentiment
```

Selecting one of these tasks is a required argument; the scraping and fine-tuning tasks have additional arguments that must be passed to successfully run them.

### Reddit scraping

This project uses the Python Reddit API Wrapper (PRAW) to scrape specified subreddits according to a desired keyword query. When performing the scraping task, a subreddit and query must be passed to the program. This task uses functions from data_files.py.

### Fine-tuning BERT

Once data has been collected from the specified subreddit(s) and queries, the fine-tuning flag concatenates individual data files into a single data file and removes duplicates. This file is then preprocessed with NLTK and the Emoji library; this data is then passed to the BERT tokenizer for encoding, and the model is fine-tuned on this encoded data. For this task, a directory containing the individual .tsv files from the scraping task must be passed. This task uses functions from data_files.py and fine_tuning.py.

### Sentiment analysis

After the model has been fine-tuned on the Reddit data, the Winoqueer data is parsed to extract the pairwise sentences, and text classification pipelines are instantiated for sentiment analysis on this data. The label predictions from the off-the-shelf and fine-tuned models are compared in a confusion matrix. This task uses functions from data_files.py and sentiment_analysis.py.

### Workflow visualization

![Workflow visualization flowchart](./LING%20742_workflow_finalized.jpeg)

## Data

### Fine-tuning dataset

The Reddit data used for this project comes from scraping both the /trans and /NonBinary subreddits for 23 salient keywords across emotions, political issues, and everyday life. In total, 7,394 posts containing titles, bodies, and comment threads were collected (3,574 from /trans, 3,820 from /NonBinary).

### Evaluation dataset

[Felkner et al.'s (2022) Winoqueer dataset](https://github.com/katyfelkner/winoqueer) contains 5,872 sentence pairs split between qualitative descriptions and pairwise comparisons. Each sentence pair contains a more stereotypical/straight and less stereotypical/queer sentence. Felkner et al.'s original task was to see which of the sentences was predicted as more likely by BERT as a way to evaluate bias; we use the dataset to compare sentiment label predictions and confidence scores between an off-the-shelf and fine-tuned BERT model.


## Discussion

The preliminary results from this project suggest that fine-tuning a large language model on domain-specific language—even on a relatively small amount of data—can help mitigate bias against marginalized communities, corroborating Felkner et al.'s study.

### Limitations
- Selection of key terms: The selection of key terms was limited in number and did not include inflectional forms, potentially affecting the accuracy of the sentiment analysis.
- Only two subreddits: The study was limited to two subreddits (/trans and /NonBinary), which may not be representative of the broader non-binary and transgender community.
- Text preprocessing: The method of data collection (i.e., comments divorced from posts), including our choice not to stem or lemmatize the data, as well as our use of the NLTK Tokenizer, may have impacted the accuracy of the sentiment analysis results.
- Small sample size: The dataset used for the project was relatively small, which may not have been sufficient for the fine-tuning process. Furthermore, the relatively small data set may limit the overall generalizability of the findings.
- Unlabeled fine-tuning and evaluation data: Due to practical constraints, our data was unlabeled; labeling the data may yield a more effective fine-tuning process and would have allowed for evaluation against gold labels rather than comparing model predictions.
- Computational processing capabilities: Due to our limited computational processing capabilities on our own devices, we used Google Colab Pro to access GPU and were not able to save much of the procedures on our own local desktops, which could have affected the consistency of the resulting output.

## Acknowledgments
Felkner, V. K., Chang, H-C. H., Jang, E., & May, J. (2022). Towards Winoqueer: Developing a benchmark for anti-queer bias in large language models. arXiv preprint arXiv:2206.11484.
