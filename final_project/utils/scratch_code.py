from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load pre-trained BERT model
model = BertModel.from_pretrained('bert-base-uncased',
                                  output_attentions=False,
                                  output_hidden_states=False)

# Define input text
text = "I really enjoyed this movie. The acting was great and the plot kept me engaged throughout."

# Tokenize input text
inputs = tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt')

# Pass input through the model
outputs = model(inputs['input_ids'], inputs['attention_mask'])

# Extract the final hidden state of the [CLS] token as the sentence embedding
sentence_embedding = outputs[0][:, 0, :]

# Perform binary classification using a linear layer on top of the sentence embedding
linear_layer = torch.nn.Linear(sentence_embedding.shape[1], 1)
classification_output = torch.sigmoid(linear_layer(sentence_embedding))

# Print the predicted sentiment
if classification_output.item() > 0.5:
    print("Positive sentiment")
else:
    print("Negative sentiment")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description=
    #                                  "Code to create a sentiment analysis pipeline with an off-the-shelf and a fine-tuned model "
    #                                              )
    # parser.add_argument("-f", "--indir", required=True, help="Data directory")
    # args = parser.parse_args()

    rows = extract_sentences(data)
    ots_sentiment_pipeline = create_ots_pipeline(id2label, label2id)
    ft_sentiment_pipeline = create_ft_pipeline(id2label, label2id)
    ft_sentiment_label_array, ots_sentiment_label_array, ft_results, ots_results = run_pipelines(ots_sentiment_pipeline,
                                                                                                 ft_sentiment_pipeline,
                                                                                                 rows)
    ots_sentiment_id_array, ft_sentiment_id_array = label_index_array(ots_sentiment_label_array,
                                                                      ft_sentiment_label_array)
    nlp_cmap = create_color_map()
    plot_confusion_matrix(ft_sentiment_id_array, ots_sentiment_id_array, nlp_cmap)
    print(plot_confusion_matrix)
