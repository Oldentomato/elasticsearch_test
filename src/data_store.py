from haystack.nodes import EmbeddingRetriever


def data_store(document_store, dfs, is_embedding=False):
    for split, df in dfs.items():
        #중복된 데이터를 제거하는 작업
        docs = [{"content": row["context"], #버전이 바뀌면서 text에서 content로 바뀜
                "meta": {"item_id": row["title"], "question_id": row["id"], "split": split}}
                for _,row in df.drop_duplicates(subset="context").iterrows()]
        document_store.write_documents(docs, index="document")

    if is_embedding:
        retriever = EmbeddingRetriever(document_store=document_store, embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1")
        document_store.update_embeddings(retriever)

    print(f"{document_store.get_document_count()}개 문서가 저장되었습니다.")

