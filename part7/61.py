from gensim.models import KeyedVectors


model = KeyedVectors.load_word2vec_format(
    './data/GoogleNews-vectors-negative300.bin.gz',
    binary=True
)

print(model.similarity('United_States', 'U.S.'))

# from gensim.models import Word2Vec
# import gensim
# model = gensim.models.Word2Vec.load(fpath)
# print(model.wv.similarity('United_States', 'U.S.'))