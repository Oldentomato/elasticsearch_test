from haystack.nodes import EmbeddingRetriever


def dataset_store(store, index_name, dfs, is_embedding=False):
    for split, df in dfs.items():
        #중복된 데이터를 제거하는 작업
        docs = [{"content": row["context"], #버전이 바뀌면서 text에서 content로 바뀜
                "meta": {"item_id": row["title"], "question_id": row["id"], "split": split}}
                for _,row in df.drop_duplicates(subset="context").iterrows()]
        store.write_documents(docs, index=index_name)#document

    if is_embedding:
        retriever = EmbeddingRetriever(document_store=store, embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1")
        store.update_embeddings(retriever)

    print(f"{store.get_document_count()}개 문서가 저장되었습니다.")


def pdfdata_store(store, index_name, dfs, is_embedding=False):
    store.write_documents(dfs, index=index_name)

    print(f"{store.get_document_count()}개 문서가 저장되었습니다.")


def delete_data(store, value):
    store.delete_documents(by_ids=[value])
    print(f"{value} is deleted")

