from transformers import T5Tokenizer


def get_tokenizer(tokenizer_path, num_bins=0):
    if 't5' in tokenizer_path:
        tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
        if num_bins:
            new_tokens = ["<time=" + str(i) + ">" for i in range(num_bins)]
            tokenizer.add_tokens(list(new_tokens))
    else:
        raise NotImplementedError(tokenizer_path)
    return tokenizer
