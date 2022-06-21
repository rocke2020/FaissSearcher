from searcher.bert_encoder import BertEncoder
import pandas as pd
from searcher.faiss_searcher import FaissSearcher


encoder = BertEncoder(config_path, checkpoint_path, dict_path)  # 预训练权重自己得准备好，也可以是自己写的encoder，必须有encode方法，建议直接继承base_encoder类来写
items = pd.read_csv(item_path)  # 候选文本集合csv文件，需要自备，需要df第一列是候选文本，其他列会在检索时自动带出。
index_param = 'HNSW64'
measurement = 'cos'

# 接下来就开始
searcher = FaissSearcher(encoder, items, index_param, measurement)
# 构建index
searcher.train()
# 保存index，方便下次调用
searcher.save_index('demo.index')
# 搜索，以文本为例
target = ['你好我叫小鲨鱼', '你好我是小兔子', '很高兴认识你']
df_res = searcher.search(target, topK=10)  # df_res即为结果