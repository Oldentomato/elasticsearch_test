# from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore #haystack v2 haystack-elasticsearch
from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore # v1 farm-haystack[elasticsearch]
# from haystack.nodes import BM25Retriever
from get_datas import get_dataset, get_pdfdata
from data_store import dataset_store, delete_data, pdfdata_store
from search_module import Embedding_Search
# from retrieve_eval import *


#https://github.com/rickiepark/nlp-with-transformers/blob/main/07_question-answering.ipynb


if __name__ == "__main__":


    n_answers = 3

    document_store = ElasticsearchDocumentStore(return_embedding=True)

    # dfs = get_dataset(dataset_name="subjqa",subset="electronics")# 데이터 로드
    # dfs = get_pdfdata("./data/")
    # dataset_store(document_store, "document", dfs, is_embedding=False) # 데이터 스토어에 저장
    # pdfdata_store(document_store, "pdf_document", dfs)

    # delete_data(document_store, "f31159ad469fea3d5249d5ca7dae7177")

    search = Embedding_Search(document_store, n_answers, "bm25")

    preds = search.run("Is it good for reading?") # 교통신호기 관리대장의 작성 요령은?Is it good for reading?

    print(f"질문: {preds['query']} \n")
    for idx in range(n_answers):
        print(f"답변 {idx+1}: {preds['answers'][idx].answer}")
        print(f"해당 리뷰 텍스트: ...{preds['answers'][idx].context}...")
        print("\n\n")


    #deprecated
    # retriever = BM25Retriever(document_store=document_store)
    # pipe = EvalRetrieverPipeline()
    # labels = set_evaldataset(dfs)
    # eval_datastore(document_store,labels)

    # run_pipeline(pipe, top_k_retriever=3)
    # print(f"재현율@3 {pipe.eval_retriever.recall: .2f}")
