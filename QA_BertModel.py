import os
import time
from haystack.nodes import EmbeddingRetriever
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.pipelines import FAQPipeline
from haystack.utils import print_answers
import pandas as pd

time.sleep(30)

host = os.environ.get("ELASTICSEARCH_HOST", "localhost")

document_store = ElasticsearchDocumentStore(
    host=host,
    username="",
    password="",
    index="document",
    embedding_field="question_emb",
    embedding_dim=384,
    excluded_meta_data=["question_emb"],
    similarity="cosine",
)

retriever = EmbeddingRetriever(
    # document_store=document_store,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    use_gpu=False,
    scale_score=False,
)

doc_dir = "data/tutorial4"
df = pd.read_csv(f"answers.csv")
# Minimal cleaning
df.fillna(value="", inplace=True)
df["question"] = df["question"].apply(lambda x: x.strip())
questions = list(df["question"].values)
df["question_emb"] = retriever.embed_queries(queries=questions).tolist()
df = df.rename(columns={"question": "content"})
df = df.rename(columns={"text": "answer"})
print(df.head())

# Convert Dataframe to list of dicts and index them in our DocumentStore
docs_to_index = df.to_dict(orient="records")
print(docs_to_index)
document_store.write_documents(docs_to_index)

pipe = FAQPipeline(retriever=retriever)

prediction = pipe.run(query="How to contact IT support?", params={"Retriever": {"top_k": 10}})
# prediction = pipe.run(query="How to contact IT support?")
print_answers(prediction, details="medium")