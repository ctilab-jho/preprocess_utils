from keras import layers
from keras.utils import pad_sequences
import tensorflow as tf
import pickle

def load_vocab(vocab_path:str):
    with open(vocab_path,'rb') as f:
        vocab = pickle.load(f)
        vocab = sorted(set(vocab))  # 중복된 단어 토큰을 제거합니다.
    return vocab

def idf(vocab:list, docs:list):
    word_counter = layers.TextVectorization(vocabulary=vocab, output_mode="count")
    df = tf.zeros(len(vocab) + 1)   # word_counter에서 자동으로 vocab에 <UNK> 토큰을 추가하므로 +1
    for doc in docs:
        word_counts = word_counter(doc)
        df += tf.cast(word_counts > 0, tf.float32)
    idf = tf.math.log(float(len(docs)) / (df + 1)) + 1.0
    return idf[1:]  # <UNK>에 대한 IDF 제거

def integer_encode_docs(vocab_path:str, docs:list, max_tokens:int=4000, max_length:int=500, num_parallel_calls=12):
    """
    주어진 단어장을 활용하여 문장의 각 단어를 대응되는 정수로 치환합니다.
    이 때 패딩(PAD)은 0, 미등록단어(UNK)는 1로 치환됩니다.
    """
    vocab = load_vocab(vocab_path)
    text_vectorization = layers.TextVectorization(
        vocabulary=vocab,
        max_tokens=max_tokens,
        output_sequence_length=max_length,
        output_mode="int",
    )
    encoded = tf.data.Dataset.from_tensor_slices(docs)
    encoded = encoded.map(lambda v : text_vectorization(v), num_parallel_calls)
    encoded= pad_sequences(encoded, maxlen=max_length)
    return encoded

def tf_idf_encode_docs(vocab_path:str, docs:list, max_tokens:int=4000, num_parallel_calls=12):
    vocab = load_vocab(vocab_path)
    text_vectorization = layers.TextVectorization(
        vocabulary=vocab,
        output_mode="tf_idf",
        idf_weights=idf(vocab, docs)
    )
    encoded = tf.data.Dataset.from_tensor_slices(docs)
    encoded = encoded.map(
        lambda v: text_vectorization(v), 
        num_parallel_calls=num_parallel_calls
    )
    return encoded

def embed_docs(vocab_path:str, docs:list, max_tokens:int=4000, max_length:int=500, embedding_dim=100):
    encoded = integer_encode_docs(vocab_path, docs, max_tokens, max_length)
    embedding = layers.Embedding(
        input_dim=max_tokens,
        output_dim=embedding_dim,
        mask_zero=True
    )
    vectors = embedding(encoded)
    return vectors



