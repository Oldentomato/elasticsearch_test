from haystack.nodes import EmbeddingRetriever, FARMReader, BM25Retriever
from haystack.pipelines import ExtractiveQAPipeline

class Embedding_Search():
    def __init__(self, store, n_answers, retrieve_mode="bm25"):
        self.n_answers = n_answers
        if retrieve_mode == "bm25":
            retriever = BM25Retriever(document_store=store)
        elif retrieve_mode == "embedding":
            #EmbeddingRetriever로 하면 저장된 데이터들을 기반으로 임베딩모델을 업데이트시켜야함(여기가 오래걸림)
            retriever = EmbeddingRetriever(document_store=store, embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1")

        

        max_seq_len, doc_stride = 384, 128
        #reader: 검색된 context와 질문을 이용하여 질문에 따른 답변을 생성해주는 역할
        reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", max_seq_len=max_seq_len,
                            doc_stride=doc_stride, return_no_answer=True)

        self.pipe = ExtractiveQAPipeline(reader, retriever)

    def run(self, query):
        preds = self.pipe.run(query=query, params={"Retriever": {"top_k": 3}, "Reader": {"top_k": self.n_answers}})

        return preds
