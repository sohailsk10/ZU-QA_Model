# from transformers import KeyBERTTokenizer
# from transformers import BertTokenizerFast, BertModel

# # Load the tokenizer
# tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# # Add the word "About" to the tokenizer
# tokenizer.add_tokens(["About"])

# # Add the word "University" to the tokenizer
# tokenizer.add_tokens(["University"])

# # Save the updated tokenizer to disk
# # tokenizer.save_pretrained("/path/to/updated_tokenizer")

# tokenizer.save_pretrained("updated_tokenizer.json", format='json')
# from transformers import BertTokenizerFast, BertModel
# import torch
# from torch import nn

# tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
# tokenizer.tokenize("[CLS] Hello world, how are you?")

# from transformers import KeyBertTokenizer

# # Load the tokenizer
# tokenizer = KeyBertTokenizer.from_pretrained("all-mpnet/base-v2")

# # Add the word "About" to the tokenizer
# tokenizer.add_tokens(["About"])

# # Add the word "University" to the tokenizer
# tokenizer.add_tokens(["University"])

# # Save the updated tokenizer to disk
# tokenizer.save_pretrained("bert_pretrained.json", format = 'json')

from keyphrase_vectorizers import KeyphraseCountVectorizer

docs = ["About the Univeristy"]
vectorizer = KeyphraseCountVectorizer()

# Print parameters
print(vectorizer.get_params())

document_keyphrase_matrix = vectorizer.fit_transform(docs).toarray()
print("document_keyphrase_matrix", document_keyphrase_matrix)

# After learning the keyphrases, they can be returned.
keyphrases = vectorizer.get_feature_names_out()
print("keyphrases", keyphrases)

# import nltk

# string = "About the University"

# # Tokenize the string into individual words
# tokens = nltk.word_tokenize(string)

# # Get all trigrams
# trigrams = nltk.ngrams(tokens, 3)

# # Convert the trigrams back to strings and join them with spaces
# trigram_strings = [' '.join(trigram) for trigram in trigrams]

# # Print the trigram strings
# print(trigram_strings)