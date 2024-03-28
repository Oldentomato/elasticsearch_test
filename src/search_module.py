from haystack.nodes import EmbeddingRetriever, FARMReader, BM25Retriever
from haystack.pipelines import ExtractiveQAPipeline, DocumentSearchPipeline
from custom_pipeline import Ensemble_Pipeline, Decision_Pipeline

class Embedding_Search():
    def __init__(self, store, n_answers, retrieve_mode="bm25", search_mode="extract"):
        self.n_answers = n_answers
        self.search_mode = search_mode
        bm_retriever = BM25Retriever(document_store=store)
        embed_retriever = EmbeddingRetriever(document_store=store, embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1")

        

        max_seq_len, doc_stride = 384, 128
        #reader: 검색된 context와 질문을 이용하여 질문에 따른 답변을 생성해주는 역할
        reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", max_seq_len=max_seq_len,
                            doc_stride=doc_stride, return_no_answer=True)

        if search_mode == "extract":
            self.pipe = ExtractiveQAPipeline(reader, bm_retriever if retrieve_mode == "bm25" else embed_retriever)
        elif search_mode == "search":
            self.pipe = DocumentSearchPipeline(bm_retriever if retrieve_mode == "bm25" else embed_retriever)
        elif search_mode == "Ensemble":
            self.pipe = Ensemble_Pipeline(bm_retriever, embed_retriever, reader)
        elif search_mode == "Decision":
            self.pipe = Decision_Pipeline(bm_retriever, embed_retriever, reader)

    def run(self, query):
        if self.search_mode == "Ensemble" or self.search_mode == "Decision":
            params = {"EmbeddingRetriever": {"top_k": 5}, "BM25Retriever": {"top_k": 5}}
        else:
            params = {"Retriever": {"top_k": 3}, "Reader": {"top_k": self.n_answers}}
        preds = self.pipe.run(query=query, params=params)

        return preds
