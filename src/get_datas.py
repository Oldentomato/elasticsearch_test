from datasets import load_dataset
from haystack.nodes import PreProcessor, TransformersDocumentClassifier
from haystack.utils import convert_files_to_docs


#dataset_url: https://huggingface.co/datasets/subjqa

def get_dataset(dataset_name, subset):
    subjqa = load_dataset(dataset_name, name=subset)

    dfs = {split: dset.to_pandas() for split, dset in subjqa.flatten().items()}

    for split, df in dfs.items():
        print(f"{split}에 있는 질문 개수: {df['id'].nunique()}")

    qa_cols = ["title", "question", "answers.text", "answers.answer_start", "context"]
    sample_df = dfs["train"][qa_cols].sample(2, random_state=7)

    print(sample_df) # debugging

    return dfs


def get_pdfdata(doc_dirs):
    all_docs = convert_files_to_docs(dir_path=doc_dirs)
    preprocessor_sliding_window = PreProcessor(split_overlap=3, split_length=10, split_respect_sentence_boundary=False)
    docs_sliding_window = preprocessor_sliding_window.process(all_docs)

    # #문서별 클래스를 제로샷 모델을 활용하여 분류하도록하려면 아래과 같이 작성
    # doc_classifier = TransformersDocumentClassifier(
    #     model_name_or_path="cross-encoder/nli-distilroberta-base",
    #     task="zero-shot-classification",
    #     labels=["labels"],
    #     batch_size=16
    # )

    # classified_docs = doc_classifier.predict(docs_sliding_window)

    # #debugging
    # print(classified_docs[0].to_dict())

    return docs_sliding_window





