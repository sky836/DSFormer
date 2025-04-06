import node2vec
import numpy as np
import networkx as nx
from gensim.models import Word2Vec

is_directed = True
p = 2
q = 1
num_walks = 100
walk_length = 80
dimensions = 64
window_size = 10
iter = 1000
Adj_file = '../../datasets/NYCTaxi/NYCTaxiadj.txt'
SE_file = '../../datasets/pems03SE.txt'

def read_graph(edgelist):
    G = nx.read_edgelist(
        edgelist, nodetype=int, data=(('weight',float),),
        create_using=nx.DiGraph())

    return G

def learn_embeddings(walks, dimensions, output_file):
    # 将节点ID转换为字符串
    walks = [list(map(str, walk)) for walk in walks]

    # 初始化 Word2Vec 模型
    model = Word2Vec(vector_size=dimensions, window=window_size, min_count=0, sg=1, workers=8)

    # 构建词汇表
    model.build_vocab(walks)  # 先构建词汇表

    # 打印词汇表的大小，确保词汇表已被正确构建
    print("Vocabulary size:", len(model.wv))

    # 训练模型
    model.train(walks, total_examples=model.corpus_count, epochs=iter)  # 然后训练模型

    # 保存嵌入向量
    model.wv.save_word2vec_format(output_file)
    return
    
nx_G = read_graph(Adj_file)
G = node2vec.Graph(nx_G, is_directed, p, q)
G.preprocess_transition_probs()
walks = G.simulate_walks(num_walks, walk_length)
learn_embeddings(walks, dimensions, SE_file)
