from preprocess_utils.word_representation import integer_encode_docs, tf_idf_encode_docs

docs = [
   "I know you have a good life about",
   "We see big things in the world, about",
   "The man and woman make it work",
   "People go from here to there",
   "I say the first day is long",
]

# print(integer_encode_docs("sample_vocab.pkl", docs, max_tokens=100, max_length=10))
print(tf_idf_encode_docs("sample_vocab.pkl", docs, max_tokens=100))