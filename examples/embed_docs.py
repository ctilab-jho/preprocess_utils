from preprocess_utils.word_representation import embed_docs

docs = [
   "I know you have a good life",
   "We see big things in the world",
   "The man and woman make it work",
   "People go from here to there",
   "I say the first day is long"
]

print(embed_docs("sample_vocab.pkl", docs, max_tokens=100, max_length=10, embedding_dim=3))