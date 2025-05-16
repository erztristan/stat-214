import torch
from transformers import BertTokenizerFast, BertModel
import numpy as np
import preprocessing as ppc

def get_embeddings(model, sentences, max_len = 128, tokenizer = None, device = 'cuda'):
    """
    Get embeddings from the model.
    
    Args:
        model: Trained BERT model
        sentences: input data
        max_length: should match "max_length" of the tokenizer used in training session
        device:（'cpu' or 'cuda'）
    
    Returns:
        embeddings: list of [tokens, embeddings] (dimension: (size_of_sentences, 2))
    """

    model.to(device)
    model.eval()

    if tokenizer is None:
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    inputs = tokenizer(
                sentences,
                is_split_into_words=True,
                return_overflowing_tokens=True,
                padding = "max_length",
                truncation = True,
                max_length = max_len,
                return_tensors = "pt",
                return_offsets_mapping=True
            )

    embeddings = []

    for i in range(len(inputs["input_ids"])):
        with torch.no_grad():
            _, hidden_states = model(inputs["input_ids"][i].unsqueeze(0), 
                                     inputs["token_type_ids"][i].unsqueeze(0), 
                                     inputs["attention_mask"][i].unsqueeze(0))
            hidden_states = hidden_states.cpu().numpy()

        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][i])
        embeddings.append([tokens, hidden_states[0], inputs.word_ids(batch_index = i)])

    return embeddings

def dict_downsample(data, dict_embed):
    keys = list(dict_embed.keys())
    dict_ds = ppc.downsample_word_vectors(keys, dict_embed, data)
    for key in keys:
        dict_ds[key] = dict_ds[key][5:-10, :]
    return dict_ds

def dict_makedelayed(dict_embed, delays):
    keys = list(dict_embed.keys())
    for key in keys:
        dict_embed[key] = ppc.make_delayed(dict_embed[key], delays)
    return dict_embed

def auto_embeddings(trained_model, data, stories, max_len = 128, tokenizer = None, device = 'cuda', 
                    delay = [0, 1, 2, 3, 4]):

    dict_embeddings = {}

    for story in stories:
        text = data[story].data
        hidden_state = get_embeddings(trained_model, text, max_len = max_len, 
                                        tokenizer = tokenizer, device = device)
        embed = np.zeros((len(text), hidden_state[0][1].shape[1]))
        count = np.zeros(len(text))

        for i in range(len(hidden_state)):
            for j in range(len(hidden_state[i][0])):
                idx = hidden_state[i][2][j]
                if idx is not None:
                    embed[idx, :] += hidden_state[i][1][j, :]
                    count[idx] += 1
                
        embed = np.where(count[:, np.newaxis] != 0, embed / count[:, np.newaxis], 0)
        dict_embeddings[story] = embed


    dict_embeddings = dict_downsample(data, dict_embeddings)
    dict_embeddings = dict_makedelayed(dict_embeddings, delay)

    return dict_embeddings


def auto_embeddings_pretrained(data, stories, delay = [0, 1, 2, 3, 4]):

    dict_embeddings = {}
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained("bert-base-uncased")

    for story in stories:
        text = data[story].data
        encoded_input = tokenizer(text, is_split_into_words=True, return_overflowing_tokens=True,
                            return_token_type_ids=True, padding=True, truncation=True, return_tensors='pt')

        encoded_input.pop('overflow_to_sample_mapping', None)

        with torch.no_grad():
            output = model(**encoded_input)
        
        hidden_state = output.last_hidden_state.cpu().numpy()
        embed = np.zeros((len(text), len(hidden_state[0][0])))
        count = np.zeros(len(text))

        for i in range(len(hidden_state)):
            word_ids = encoded_input.word_ids(batch_index = i)
            for j in range(len(hidden_state[i])):
                idx = word_ids[j]
                if idx is not None:
                    embed[idx, :] += hidden_state[i][j, :]
                    count[idx] += 1
                
        embed = np.where(count[:, np.newaxis] != 0, embed / count[:, np.newaxis], 0)
        dict_embeddings[story] = embed


    dict_embeddings = dict_downsample(data, dict_embeddings)
    dict_embeddings = dict_makedelayed(dict_embeddings, delay)

    return dict_embeddings


def auto_embeddings_finetuned(data, model, tokenizer, stories, delay = [1, 2, 3, 4]):

    dict_embeddings = {}

    for story in stories:
        text = data[story].data
        encoded_input = tokenizer(text, is_split_into_words=True, return_overflowing_tokens=True,
                            return_token_type_ids=True, padding=True, truncation=True, return_tensors='pt')
        
        inputs = {
            "input_ids": encoded_input["input_ids"],
            "attention_mask": encoded_input["attention_mask"],
        }

        with torch.no_grad():
            output = model.bert(**inputs)
        
        hidden_state = output.last_hidden_state.cpu().numpy()
        embed = np.zeros((len(text), len(hidden_state[0][0])))
        count = np.zeros(len(text))

        for i in range(len(hidden_state)):
            word_ids = encoded_input.word_ids(batch_index = i)
            for j in range(len(hidden_state[i])):
                idx = word_ids[j]
                if idx is not None:
                    embed[idx, :] += hidden_state[i][j, :]
                    count[idx] += 1
                
        embed = np.where(count[:, np.newaxis] != 0, embed / count[:, np.newaxis], 0)
        dict_embeddings[story] = embed


    dict_embeddings = dict_downsample(data, dict_embeddings)
    dict_embeddings = dict_makedelayed(dict_embeddings, delay)

    return dict_embeddings