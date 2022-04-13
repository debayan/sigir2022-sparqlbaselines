import pickle

file=open('ent_rels.pickle','rb')
data=pickle.load(file)
file.close()

from elasticsearch import Elasticsearch
es = Elasticsearch(host="localhost",port=9700)
es9800 = Elasticsearch(host="localhost",port=9800)

for i,lis in enumerate(data):
    for j,item in enumerate(lis):
        try:
            res = es9800.search(index="dbpedialabelindex01", body={"query":{"term":{"uri":{"value":item[1:-1]}}}})
            data[i][j]=res['hits']['hits'][0]['_source']['dbpediaLabel']
        except :
            data[i][j]='<extra_id_17>'

print(data[0])

file=open('labels.pickle','wb')
pickle.dump(data,file)
file.close()