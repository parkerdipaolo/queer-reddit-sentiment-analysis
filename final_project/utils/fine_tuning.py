import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import BertTokenizer, BertForMaskedLM, AdamW


def finetune_bert(df):
    """
    Code for fine-tuning BERT with Reddit data
    :param df: dataframe containing preprocessed Reddit data
    :return: fine-tuned ('bert-unsupervised') language model
    """
    # load the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # tokenize the text data
    encoded_data = tokenizer.batch_encode_plus(
        df.text.values,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=512,
        return_tensors='pt'
    )

    # create the input tensors
    input_ids = encoded_data['input_ids']
    attention_masks = encoded_data['attention_mask']

    # create the dataset
    dataset = TensorDataset(input_ids, attention_masks)

    # set the device to run the model on
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load the BERT model with masked language modeling head
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    model.to(device)

    # set the optimizer and learning rate
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # create the data loader
    batch_size = 32
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    # train the model
    epochs = 4
    for epoch in range(epochs):
        # training loop
        total_loss = 0
        model.train()
        for step, batch in enumerate(dataloader):
            batch_input_ids = batch[0].to(device)
            batch_attention_masks = batch[1].to(device)
            batch_masked_ids = batch_input_ids.clone()
            masked_indices = torch.where(batch_input_ids != tokenizer.pad_token_id)
            masked_indices = torch.stack([masked_indices[0], masked_indices[1]], dim=1)
            num_masks = int(0.15 * masked_indices.shape[0])
            rand_indices = torch.randperm(masked_indices.shape[0])[:num_masks]
            masked_indices = masked_indices[rand_indices, :]
            batch_masked_ids[masked_indices[:, 0], masked_indices[:, 1]] = tokenizer.mask_token_id
            model.zero_grad()
            outputs = model(
                batch_masked_ids,
                attention_mask=batch_attention_masks,
                labels=batch_input_ids
            )
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # print the average loss after each epoch
        avg_train_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1} - Average loss: {avg_train_loss:.2f}')

    # save the fine-tuned model
    model.save_pretrained('bert-unsupervised')
