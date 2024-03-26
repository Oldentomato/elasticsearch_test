from haystack.pipeline import Pipeline
from haystack.eval import EvalDocuments
from haystack import Label

#각 retrive방식(임베딩이나 bm25 등)에 따른 성능평가를 위한 코드
class EvalRetrieverPipeline:
    def __init__(self, retriever):
        self.retriever = retriever
        self.eval_retriever = EvalDocuments()
        pipe = Pipeline()
        pipe.add_node(component=self.retriever, name="ESRetriever",
                      inputs=["Query"])
        pipe.add_node(component=self.eval_retriever, name="EvalRetriever",
                      inputs=["ESRetriever"])
        
        self.pipeline = pipe


def set_evaldataset(dfs):
    labels = []
    for i, row in dfs["test"].iterrows():
        #리트리버에서 필터링을 위해 사용하는 메타데이터
        meta = {"item_id": row["title"], "question_id": row["id"]}
        #답이 있는 질문을 레이블에 추가
        if len(row["answers.text"]):
            for answer in row["answers.text"]:
                label = Label(
                    question=row["question"], answer=answer, id=i, origin=row["id"],
                    meta=meta, is_correct_answer=True, is_correct_document=True,
                    no_answer=False
                )
                labels.append(label)

        # 답이 없는 질문을 레이블에 추가합니다
        else:
            label=Label(
                question=row["question"], answer="", id=i, origin=row["id"],
                meta=meta, is_correct_answer=True, is_correct_document=True,
                no_answer=True
            )
            labels.append(label)

    #debugging
    print(labels[0])

    return labels

def eval_datastore(store, labels):
    store.write_labels(labels, index="label")
    print(f"""{store.get_label_count(index="label")}개의 \
          질문 답변 쌍을 로드했습니다. """)
    
    labels_agg = store.get_all_labels_aggregated(
        index="label",
        open_domain=True,
        aggregate_by_meta=["item_id"]
    )

    print(len(labels_agg))

    return labels_agg

def run_pipeline(pipeline, labels_agg, top_k_retriever=10, top_k_reader=4):
    for l in labels_agg:
        _ = pipeline.pipeline.run(
            query=l.question,
            
        )

