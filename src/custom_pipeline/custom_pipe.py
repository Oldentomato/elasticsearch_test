from haystack.nodes import JoinDocuments
from haystack.pipelines import Pipeline
from haystack import BaseComponent

#참고: https://haystack.deepset.ai/tutorials/11_pipelines

class CustomQueryClassifier(BaseComponent):

    def run(self, query):
        if "?" in query:
            return {}, "output_2",
        else:
            return {}, "output_1"

    def run_batch(self, queries):
        split = {"output_1": {"queries": []}, "output_2": {"queries": []}}
        for query in queries:
            if "?" in query:
                split["output_2"]["queries"].append(query)
            else:
                split["output_1"]["queries"].append(query)

        return split, "split"
    


#bm25와 embedding의 결과를 결합
def Ensemble_Pipeline(bm25_retriever, embedding_retriver, reader):
    p_ensemble = Pipeline()
    p_ensemble.add_node(
        component=bm25_retriever,
        name="BM25Retriever",
        inputs=["Query"]
    )
    p_ensemble.add_node(
        component=embedding_retriver, 
        name="EmbeddingRetriever",
        inputs=["Query"]
    )
    p_ensemble.add_node(
        component=JoinDocuments(join_mode="concatenate"), name="JoinResults", inputs=["BM25Retriever", "EmbeddingRetriever"]
    )
    p_ensemble.add_node(component=reader, name="Reader", inputs=["JoinResults"])

    return p_ensemble


#쿼리의 형태에 따라 bm25혹은 embedding으로 연결됨(현재는 쿼리에 ?의 여부에 따름)
def Decision_Pipeline(bm25_retriever, embedding_retriever, reader):
    p_classifier = Pipeline()
    p_classifier.add_node(component=CustomQueryClassifier(), name="QueryClassifier", inputs=["Query"])
    p_classifier.add_node(component=bm25_retriever, name="BM25Retriever", inputs=["QueryClassifier.output_1"])
    p_classifier.add_node(component=embedding_retriever, name="EmbeddingRetriever", inputs=["QueryClassifier.output_2"])
    p_classifier.add_node(component=reader, name="QAReader", inputs=["BM25Retriever","EmbeddingRetriever"])

    return p_classifier