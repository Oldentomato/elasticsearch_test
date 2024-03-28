from haystack.utils import fetch_archive_from_http
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import PreProcessor, EmbeddingRetriever, FARMReader, BM25Retriever
# from custom_pipeline import Ensemble_Pipeline, Decision_Pipeline
from haystack.pipelines import ExtractiveQAPipeline, DocumentSearchPipeline
from haystack.schema import EvaluationResult, MultiLabel

#참고 https://haystack.deepset.ai/tutorials/05_evaluation

if __name__ == "__main__":

    doc_dir = "../data/tutorial5"
    s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/nq_dev_subset_v2.json.zip"
    fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

    doc_index = "tutorial5_docs"
    label_index = "tutorial5_labels"

    document_store = ElasticsearchDocumentStore(
        host="localhost",
        port=9200,
        index=doc_index,
        label_index=label_index,
        embedding_field="emb",
        embedding_dim=768,
        excluded_meta_data=["emb"]
    )

    preprocessor = PreProcessor(
        split_by="word",
        split_length=200,
        split_overlap=0,
        split_respect_sentence_boundary=False,
        clean_empty_lines=False,
        clean_whitespace=False
    )
    document_store.delete_documents(index=doc_index)
    document_store.delete_documents(index=label_index)

    document_store.add_eval_data(
        filename="../data/tutorial5/nq_dev_subset_v2.json",
        doc_index=doc_index,
        label_index=label_index,
        preprocessor=preprocessor,
    )

    retriever = BM25Retriever(document_store=document_store) # or embeddingretriever

    reader = FARMReader("deepset/roberta-base-squad2", top_k=4, return_no_answer=True)

    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever) # or custom_pipeline
    # pipeline = DocumentSearchPipeline(retriever=retriever)

    eval_labels = document_store.get_all_labels_aggregated(drop_negative_labels=True, drop_no_answers=True)

    eval_result = pipeline.eval(labels=eval_labels, params={"Retriever": {"top_k": 5}})

    eval_result.save("../result/")

    # caculate metric

    saved_eval_result = EvaluationResult.load("../result/")
    metrics = saved_eval_result.calculate_metrics()

    print(f'Retriever - Recall (single relevant document): {metrics["Retriever"]["recall_single_hit"]}')
    print(f'Retriever - Recall (multiple relevant documents): {metrics["Retriever"]["recall_multi_hit"]}')
    print(f'Retriever - Mean Reciprocal Rank: {metrics["Retriever"]["mrr"]}')
    print(f'Retriever - Precision: {metrics["Retriever"]["precision"]}')
    print(f'Retriever - Mean Average Precision: {metrics["Retriever"]["map"]}')

    print(f'Reader - F1-Score: {metrics["Reader"]["f1"]}')
    print(f'Reader - Exact Match: {metrics["Reader"]["exact_match"]}')

