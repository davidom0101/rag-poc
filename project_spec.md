## Project Specification: Multimodal RAG POC using Docling & LlamaIndex
This document defines the technical specification and step-by-step implementation guide for a Proof of Concept (POC) Retrieval-Augmented Generation (RAG) system. The pipeline parses software user manuals (text, tables, and screenshots) to evaluate layout-aware information retrieval.
------------------------------
## 1. Project Overview & Objectives

* Core Goal: Build a local RAG pipeline capable of accurately answering text-based, spatial, and tabular queries derived from complex layout PDFs (e.g., software user manuals with embedded screenshots). [1] 
* Target Stack: Docling (Layout Parsing), LlamaIndex (Orchestration & Vector Storage), Milvus Lite (Vector Database), and OpenAI's GPT-4o mini (text LLM + vision, via API key).
* Starting Document: gimp-help-2.10-en.pdf — the only document used to validate the pipeline for this POC.
* Key KPI: Successfully parse and retrieve information where structural context (headings, tables, figure-text proximity) must be preserved to answer the query.
1
------------------------------
## 2. System Architecture
1
[ PDF User Manual ] 
       │
       ▼ (Layout-Aware Structural Analysis)
┌────────────────────────────────────────┐
│ Docling Engine                         │
│ 1. OCR & Layout Object Detection       │ -> Converts to Markdown/JSON
│ 2. Semantic Chunking (Hierarchical)     │    preserving Table/Figure anchors
│ 3. Picture Extraction (generate_picture_images) │ -> raw screenshot crops
└────────────────────────────────────────┘
       │                              │
       ▼ (Node Ingestion)             ▼ (per screenshot)
┌────────────────────────────────────────┐   ┌──────────────────────────────┐
│ LlamaIndex Orchestration               │   │ 1. Save PNG -> ./extracted_images/│
│ 1. Node Creation (DoclingNodeParser)   │   │ 2. GPT-4o mini Vision captions│
│ 2. Structure-Aware Chunking            │   │    UI elements/shapes/layout  │
│    -> preserves headings/bbox/page     │   │    -> standalone TextNode     │
│ 3. Ref-match doc_items -> picture/     │   │ 3. ref (picture + native      │
│    native-caption refs, tag matching   │   │    caption) -> saved PNG path │
│    chunks with linked_image_path       │◄──┤    recorded for step 3        │
│ 4. Local Dense Embeddings Generation   │   └──────────────────────────────┘
│    -> e.g., BAAI/bge-small-en-v1.5     │
│    (TEXT ONLY -- no image embeddings)  │
└────────────────────────────────────────┘
       │
       ▼ (Vector Insertion, body + caption nodes; images stay on disk, only their
       │  file paths ride along as node metadata)
       [ Local Vector Store (Milvus Lite / Disk) -- text embeddings only ]
                            ▲
                            │ (Retrieve top-k contexts via TEXT embedding similarity)
                  ┌────────────────────────────────────────────────┐
                  │ Retrieve-then-synthesize (query_with_images)   │
                  │ 1. Text-embedding retrieval (unchanged)         │
                  │ 2. Collect linked_image_path from matched nodes │
                  │ 3. Send context text + real screenshot pixels   │
                  │    to GPT-4o mini vision (ChatMessage blocks)   │ -> Final Grounded Answer
                  └────────────────────────────────────────────────┘

------------------------------
## 3. Technical Requirements## Software Dependencies

* Python: ^3.10 or ^3.11
* Layout Parsing: docling>=2.0.0
* RAG Framework: llama-index-core, llama-index-node-parser-docling
* Embeddings: llama-index-embeddings-huggingface
* Vector Store: llama-index-vector-stores-milvus (Milvus Lite)
* LLM: llama-index-llms-openai, openai (requires OPENAI_API_KEY)
* Config: python-dotenv (loads OPENAI_API_KEY from a local .env file)

## Hardware Requirements

* CPU/GPU: A modern multi-core CPU is sufficient for Docling parsing and embedding generation. LLM inference runs via the OpenAI API, so no local GPU is required for generation.

------------------------------
## 4. Implementation Step-by-Step## Step 1: Environment Setup
Create a isolated directory and install the required library ecosystem.

# Create and enter project directory
mkdir docling-rag-poc && cd docling-rag-poc
# Setup virtual environment
python3 -m venv venv
source venv/bin/activate
# Install exact framework dependencies
pip install docling llama-index-core llama-index-node-parser-docling \
            llama-index-embeddings-huggingface llama-index-vector-stores-milvus \
            llama-index-llms-openai openai python-dotenv milvus-lite

# Note (Windows): pymilvus's "milvus_lite" extra is gated by a platform marker that
# doesn't resolve on Windows, so it silently installs nothing extra there. Installing
# the milvus-lite package explicitly (as above) is required on Windows.

# Set your OpenAI API key (required for GPT-4o mini) — create a .env file
# in the project root containing: OPENAI_API_KEY=sk-...

## Step 2: Initialize Core Pipeline (app.py)
Create a python script named app.py. This script pulls local configuration parameters directly aligned with the official Docling implementation patterns. [2]

Docling does not extract any pixel content or visual description by default (`do_picture_description` and `generate_picture_images` both default to `False`) — a picture node only carries a bounding box, so a text-only pipeline is blind to screenshots despite manuals being full of them. `app.py` extracts each screenshot to `./extracted_images/`, captions it via GPT-4o-mini vision at ingestion time (indexed as its own retrievable node alongside the body text), and links any matching text node (both the generated caption and Docling's own native figure captions, e.g. "Figure 3.4 Screenshot of the Toolbox") back to that saved file. At query time retrieval stays 100% text-embedding-driven (no CLIP, no second vector store — see [Known Issue 4](#5-known-issues) for why), but the real screenshot pixels for any matched node are sent into GPT-4o-mini's vision input alongside the retrieved text, so the final answer is grounded in a fresh look at the image rather than a lossy pre-written caption.

import base64
import io
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.schema import TextNode
from llama_index.core.llms import ChatMessage, ImageBlock, TextBlock
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.llms.openai import OpenAI
from openai import OpenAI as OpenAIClient

load_dotenv()

IMAGE_DIR = Path("./extracted_images")
MAX_SYNTHESIS_IMAGES = 3

VISION_PROMPT = (
    "This is a screenshot from a software user manual. Describe precisely what UI "
    "elements, icons, buttons, and layout are visible, including shapes, positions "
    "(e.g. top-left, first row), and colors where relevant."
)

QA_PROMPT_TEMPLATE = (
    "Context information from the manual is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and any attached screenshots, answer the query. "
    "If screenshots are attached, look at their actual pixels to answer precisely "
    "rather than relying only on the context text.\n"
    "Query: {query_str}\n"
    "Answer: "
)

def extract_and_caption_pictures(dl_doc, vision_client) -> tuple[list[TextNode], dict[str, str]]:
    """Save each screenshot to disk, caption it via GPT-4o-mini vision, and build
    a ref -> image path map so retrieved text nodes can be linked back to their
    real image file. Captions make every screenshot independently retrievable by
    text search (most GIMP screenshots have no native caption sentence); the ref
    map lets synthesis attach the real pixels instead of just the caption text."""
    IMAGE_DIR.mkdir(exist_ok=True)
    caption_nodes = []
    ref_to_image_path: dict[str, str] = {}

    for i, pic in enumerate(dl_doc.pictures):
        img = pic.get_image(dl_doc)
        if img is None:
            continue
        page_no = pic.prov[0].page_no if pic.prov else None

        image_path = str(IMAGE_DIR / f"pic_{page_no}_{i}.png")
        img.save(image_path, format="PNG")

        ref_to_image_path[pic.self_ref] = image_path
        for cap_ref in pic.captions:
            ref_to_image_path[cap_ref.cref] = image_path

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        resp = vision_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }],
            max_tokens=300,
        )
        caption = resp.choices[0].message.content
        caption_nodes.append(TextNode(
            text=f"[Screenshot on page {page_no}] {caption}",
            metadata={
                "page_no": page_no, "picture_index": i, "source": "image_caption",
                "linked_image_path": image_path,
            },
        ))

    return caption_nodes, ref_to_image_path

def link_body_nodes_to_images(body_nodes: list[TextNode], ref_to_image_path: dict[str, str]) -> None:
    """Tag body nodes with linked_image_path when they contain a doc item whose
    ref is a picture or one of its native captions. DoclingNodeParser stores each
    chunk's source doc item refs in metadata["doc_items"][*]["self_ref"], which
    lets us match chunks back to the picture they describe."""
    for node in body_nodes:
        for doc_item in node.metadata.get("doc_items", []):
            image_path = ref_to_image_path.get(doc_item.get("self_ref"))
            if image_path:
                node.metadata["linked_image_path"] = image_path
                break

def query_with_images(index: VectorStoreIndex, query_str: str, similarity_top_k: int = 3) -> str:
    """Retrieve via standard text-embedding similarity (no CLIP, no separate image
    vector store), then attach the real pixels of any linked screenshots to the
    GPT-4o-mini synthesis call."""
    nodes = index.as_retriever(similarity_top_k=similarity_top_k).retrieve(query_str)
    context_str = "\n\n".join(n.get_content() for n in nodes)

    images = []  # list of (page_no, path), de-duplicated by path
    seen_paths = set()
    for n in nodes:
        path = n.metadata.get("linked_image_path")
        if path and path not in seen_paths:
            seen_paths.add(path)
            images.append((n.metadata.get("page_no"), path))
    if len(images) > MAX_SYNTHESIS_IMAGES:
        print(f"    (dropping {len(images) - MAX_SYNTHESIS_IMAGES} linked image(s) beyond the "
              f"{MAX_SYNTHESIS_IMAGES}-image synthesis cap)")
        images = images[:MAX_SYNTHESIS_IMAGES]

    prompt = QA_PROMPT_TEMPLATE.format(context_str=context_str, query_str=query_str)
    # Label each image with its page number immediately before it so the model can
    # correlate "which screenshot is which" instead of reasoning over an anonymous
    # blob of images -- otherwise multi-image answers can latch onto the wrong one.
    blocks = []
    for page_no, path in images:
        blocks.append(TextBlock(text=f"[Screenshot from page {page_no}]"))
        blocks.append(ImageBlock(path=path))
    blocks.append(TextBlock(text=prompt))

    response = Settings.llm.chat([ChatMessage(role="user", blocks=blocks)])
    return response.message.content

def run_poc_pipeline(pdf_path: str):
    print("[1/6] Initializing Models & Settings...")

    # Configure Local Embedding Model via HuggingFace
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.embed_model = embed_model
    embed_dim = len(embed_model.get_text_embedding("hi"))

    # Configure LLM via OpenAI (requires OPENAI_API_KEY environment variable)
    Settings.llm = OpenAI(model="gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"])
    vision_client = OpenAIClient(api_key=os.environ["OPENAI_API_KEY"])

    print("[2/6] Preparing Local Storage Engine...")
    # Initialize a persistent local disk-based vector instance via Milvus Lite.
    # output_fields is required: pymilvus's search() (unlike query()) does not expand
    # the "*" wildcard against this backend, so without "text" every retrieval silently
    # comes back with empty text and the LLM answers ungrounded. "_node_content" is
    # equally required for METADATA: without it, _parse_from_milvus_results() falls
    # back to only the explicitly-requested fields, silently dropping every custom
    # metadata key (including our linked_image_path) even though "text" still works.
    vector_store = MilvusVectorStore(
        uri="./milvus_demo.db", dim=embed_dim, overwrite=True,
        output_fields=["text", "_node_content"],
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print(f"[3/6] Parsing Document Structure via Docling: {pdf_path}...")
    # generate_picture_images=True embeds each screenshot's pixel data in the
    # conversion result so it can be saved to disk and captioned below. images_scale
    # defaults to 1.0 (~72 DPI), which produces crops too small (e.g. 84x172px for a
    # toolbox screenshot) for GPT-4o-mini to read fine icon/button detail from -- 3.0
    # gives a much higher-resolution crop for accurate visual grounding.
    opts = PdfPipelineOptions()
    opts.generate_picture_images = True
    opts.images_scale = 3.0
    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    dl_doc = converter.convert(pdf_path).document

    print(f"[4/6] Extracting & captioning {len(dl_doc.pictures)} screenshots via GPT-4o-mini vision...")
    caption_nodes, ref_to_image_path = extract_and_caption_pictures(dl_doc, vision_client)

    print("[5/6] Vectorizing Hierarchy & Building Local Index...")
    # JSON export + DoclingNodeParser preserves headings, table/figure anchors,
    # and page/bbox grounding -- Markdown export alone loses this structure.
    body_document = Document(text=json.dumps(dl_doc.export_to_dict()))
    node_parser = DoclingNodeParser()
    body_nodes = node_parser.get_nodes_from_documents([body_document])
    link_body_nodes_to_images(body_nodes, ref_to_image_path)

    index = VectorStoreIndex(
        nodes=body_nodes + caption_nodes,
        storage_context=storage_context,
    )

    print("[6/6] Executing Test Query against Layout Data...")
    # Test Evaluation Prompts
    test_query = "What step-by-step UI actions or navigation paths are outlined in the settings chapter?"
    print(f"\nTarget Query: {test_query}")

    answer = query_with_images(index, test_query, similarity_top_k=3)
    print(f"\nGenerated Answer:\n{answer}")
if __name__ == "__main__":
    # Target manual drop placement
    sample_pdf = "./gimp-help-2.10-en.pdf"
    
    if not os.path.exists(sample_pdf):
        print(f"Please drop a user manual PDF at '{sample_pdf}' to run the pipeline.")
    else:
        run_poc_pipeline(sample_pdf)

## Step 3: Run and Test

   1. Place the target user manual PDF in the root folder as gimp-help-2.10-en.pdf (the only document used for this POC).
   2. Ensure the OPENAI_API_KEY environment variable is set.
   3. Execute the pipeline:
   
   python app.py
   
   Running the pipeline creates `./extracted_images/` (saved screenshot PNGs) and `./milvus_demo.db` (the vector store) in the project root — both are runtime artifacts, not source, and should be gitignored.
   
   
------------------------------
## 5. Known Issues

Five issues were found while implementing and verifying this pipeline (issues 1-2 are Windows-specific; 3-5 apply on any platform). All are silent — the pipeline appears to succeed without any of the fixes, but produces wrong, ungrounded, or poorly-retrieved results.

1. **Docling `std::bad_alloc` after ~8-9 pages per process.** A single `DocumentConverter.convert()` call reliably fails on Windows after roughly the 8th or 9th page it processes, regardless of which pages or what content — confirmed by testing multiple page ranges. It is a native-layer resource leak (unaffected by disabling OCR), not a content or memory-capacity issue (24GB RAM, only ~14GB in use). `result.status` degrades to `PARTIAL_SUCCESS`/pages silently drop out — it does not raise. **Implication:** `gimp-help-2.10-en.pdf` (1,070 pages) cannot be parsed in one process. Pages must be converted in small batches (~6 pages, tested clean), each in a fresh subprocess, since a new process resets the leaked native state. `app.py` currently ingests the whole PDF in one call and is only safe for small documents/page ranges until batched ingestion is added — see the Batched Ingestion note below.
2. **`MilvusVectorStore` search returns empty text (and drops all metadata) with an incomplete `output_fields` config.** `pymilvus`'s `search()` (unlike `query()`) does not expand the `"*"` wildcard against the Milvus Lite backend used here — every retrieved node comes back with `text=""` and empty metadata, so the LLM silently answers from its own general knowledge instead of the retrieved context. Fixed by passing `output_fields=["text", "_node_content"]` explicitly to `MilvusVectorStore(...)` (already applied in `app.py`, [Step 2](#step-2-initialize-core-pipeline-apppy)). Verified (`text` alone): without the fix, a question about GIMP's Toolbox got a generic non-answer; with it, the answer correctly cited "Edit → Preferences → Toolbox: Tools configuration" from the source text. **This bug has a second, sneakier layer:** `output_fields=["text"]` alone fixes the text but silently drops all *custom metadata* too — `llama_index/vector_stores/milvus/base.py`'s `_parse_from_milvus_results()` only reconstructs full node metadata when `"_node_content"` is present in the returned entity; with just `"text"` requested, the retrieved node's `metadata` dict comes back empty even though `text` is populated, so anything read from `node.metadata` after retrieval (e.g. our `linked_image_path` — see issue 4) silently vanishes with zero error, while the answer still looks plausible. Caught by explicitly re-querying a fresh index reconnect and printing `node.metadata.get("linked_image_path")` for the top-k results — it came back `None` for every node until `"_node_content"` was added to `output_fields`. **Always request `"_node_content"` alongside `"text"` whenever node metadata is read after retrieval, not just when the answer "looks" grounded** — a plausible-sounding answer is not proof of real grounding, especially from an LLM with strong prior knowledge of a popular tool like GIMP.
3. **Docling captures zero visual content by default, despite this being a "multimodal" spec.** `do_picture_description` and `generate_picture_images` both default to `False` — a picture node only ever carries a bounding box and a list of nearby text refs, never pixel data or a description. A question about a screenshot's contents ("what shape is the icon top-left of the toolbox?") got an honest "not provided in the available information" rather than a hallucination, but that also means it could never be answered. First fixed by enabling `generate_picture_images=True` and captioning each extracted screenshot with GPT-4o-mini vision at ingestion time (indexed as its own node) — see issue 4 below for why that alone wasn't the final design.
4. **CLIP-based dual-embedding retrieval (`MultiModalVectorStoreIndex`) is the "textbook" multimodal answer, but a poor fit for dense-UI-text screenshots.** We evaluated LlamaIndex's `MultiModalVectorStoreIndex` (confirmed against installed `llama-index-core` 0.14.23 source): it requires a CLIP embedding model (`llama-index-embeddings-clip`, not installed, ~350MB checkpoint download) and a second Milvus Lite collection, and does two independent embedding-similarity searches (text vs. image) whose results get concatenated — it does not work by "text match → fetch its linked image." Rejected because CLIP is trained on general image/text concept pairs and is notoriously bad at reading small, dense text (menu labels, button captions) — exactly what GIMP screenshots are full of; it would frequently retrieve the wrong screenshot. **Final design:** keep retrieval 100% text-embedding-driven (the one Milvus Lite collection already in place, no CLIP, no second vector store) — every screenshot is made searchable via its GPT-4o-mini-generated caption (most GIMP screenshots have no native caption sentence — confirmed 1-of-16 in testing — so this is required for retrievability, not optional), and any node whose source doc item matches a picture or its native caption ref (via `DoclingNodeParser`'s `doc_items[].self_ref` chunk metadata) is tagged with `linked_image_path`. At synthesis time, `query_with_images()` retrieves via ordinary text-embedding similarity, then sends the *real* screenshot pixels (not the caption text) for any matched node into GPT-4o-mini vision via `ImageBlock`/`ChatMessage`. This avoids CLIP's UI-text blindness — see issue 5 for the honest limit of what real-pixel synthesis actually achieves.
5. **Real-pixel synthesis is verified working, but GPT-4o-mini's fine-icon-shape answers are inconsistent even with the correct image attached — don't overclaim visual accuracy.** Two sub-findings, both confirmed directly (not inferred):
   - *Retrieval/linking engineering is correct.* Directly inspected retrieved nodes' `metadata["linked_image_path"]` after a fresh index reconnect — confirmed the top-scoring node for the toolbox-icon question resolves to the correct saved file (`pic_3_4.png`, the "Figure 3.4 Screenshot of the Toolbox" image), matching Docling's own native caption link.
   - *Image resolution matters a lot, but doesn't fully solve accuracy.* Docling's `images_scale` defaults to `1.0` (~72 DPI), producing an 84×172px crop for that screenshot — too small for GPT-4o-mini to read individual toolbox icons from. Raised to `images_scale = 3.0` (253×515px, already applied in `app.py`). Even so, re-running the *identical* top-left-icon question three times against the *same, correctly-attached* image produced three different answers: "rectangular selection tool" (correct — directly confirmed by viewing `pic_3_4.png`, the actual first icon is a dashed-rectangle Rectangle Select tool), "pen or drawing tool" (wrong), and "scissors" (wrong). **This means the pipeline's engineering (retrieval, linking, real-pixel attachment) is verified correct, but answer accuracy on fine-grained icon-shape questions is bounded by GPT-4o-mini vision's reliability on small icon-grid crops, not by anything further fixable in `app.py`.** Don't treat a single plausible-sounding answer as proof of accuracy — this class of question benefits from being asked multiple times, or from a tighter crop around just the region in question, neither of which this POC implements.

### Batched Ingestion (required for the full 1,070-page manual)
To ingest the complete document, replace the single `converter.convert(pdf_path)` call in `app.py` with a loop that, for each ~6-page batch, shells out to a small worker script (fresh `DocumentConverter()` per subprocess) and aggregates each batch's JSON export + picture captions. At ~50s/6-page batch (plus one vision API call per screenshot) this is a multi-hour one-time ingestion for the full manual. Not yet implemented in `app.py`.

------------------------------
## 6. POC Evaluation & Verification Matrix
To prove the RAG capability beyond standard keyword lookups, evaluate the pipeline performance using these criteria:

| Test Type | Objective | Expected Outcome | Pass Criteria |
|---|---|---|---|
| Tabular Verification | Ask for a metric or key-bind inside a structured table. | System reads cell intersections. | Returns exact column value matching the row target. |
| Spatial Proximity | Query a feature described directly beneath a screenshot placeholder. | System keeps markdown captions unified with parental headings. | No "hallucinated" features from nearby unrelated steps. |
| Hierarchy Navigation | Ask to list steps spanning an explicit multi-tier UI path. | Docling layout breaks preserve parent-child header strings. | Complete bullet lists generated cleanly in chronological order. |

