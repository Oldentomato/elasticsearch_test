from datasets import load_dataset

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


