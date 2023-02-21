"""Set the logging level to INFO:"""

import logging
from haystack.document_stores import InMemoryDocumentStore
from haystack.pipelines import ExtractiveQAPipeline
from haystack.pipelines.standard_pipelines import TextIndexingPipeline
import os
from haystack.nodes import BM25Retriever
from haystack.nodes import FARMReader
from haystack.utils import print_answers
from pprint import pprint

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)


document_store = InMemoryDocumentStore(use_bm25=True)
doc_dir = "ZU_data"

files_to_index = [doc_dir + "/" + f for f in os.listdir(doc_dir)]
indexing_pipeline = TextIndexingPipeline(document_store)
indexing_pipeline.run_batch(file_paths=files_to_index)

retriever = BM25Retriever(document_store=document_store)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)
pipe = ExtractiveQAPipeline(reader, retriever)


prediction = pipe.run(
    query="I am looking for Convention Center Fact Sheet in pdf?",
    params={
        "Retriever": {"top_k": 10},
        "Reader": {"top_k": 5}
    }
)
pprint(prediction)

print_answers(
    prediction,
    details="medium" ## Choose from `minimum`, `medium`, and `all`
)
