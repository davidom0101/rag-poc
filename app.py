import os

# Force pure offline mode for the local HuggingFace-hosted models (embeddings,
# Docling's layout detector + TableFormer). Must be set before any huggingface_hub/
# transformers/docling import, since they read these at import time. This does NOT
# affect GPT-4o-mini -- that's an OpenAI API call and always needs internet regardless.
# Requires the models to already be cached locally (they are, after the first run
# without these vars set) -- otherwise loading fails immediately instead of downloading.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import base64
import io
import json
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
    real image file. Docling itself never extracts pixel content or descriptions
    by default (do_picture_description and generate_picture_images both default
    to False), so without this step the pipeline is blind to screenshots despite
    claiming to be multimodal. Captions make every screenshot independently
    retrievable by text search; the ref map lets synthesis attach the real pixels
    instead of just the (lossy) caption text.
    """
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
                "page_no": page_no,
                "picture_index": i,
                "source": "image_caption",
                "linked_image_path": image_path,
            },
        ))

    return caption_nodes, ref_to_image_path


def link_body_nodes_to_images(body_nodes: list[TextNode], ref_to_image_path: dict[str, str]) -> None:
    """Tag body nodes with linked_image_path when they contain a doc item whose
    ref is a picture or one of its native captions (e.g. "Figure 3.4 Screenshot
    of the Toolbox"). DoclingNodeParser stores each chunk's source doc item refs
    in metadata["doc_items"][*]["self_ref"] (from docling_core's chunk.meta),
    which lets us match chunks back to the picture they describe."""
    for node in body_nodes:
        for doc_item in node.metadata.get("doc_items", []):
            image_path = ref_to_image_path.get(doc_item.get("self_ref"))
            if image_path:
                node.metadata["linked_image_path"] = image_path
                break


def query_with_images(index: VectorStoreIndex, query_str: str, similarity_top_k: int = 3) -> str:
    """Retrieve via standard text-embedding similarity (no CLIP, no separate image
    vector store), then attach the real pixels of any linked screenshots to the
    GPT-4o-mini synthesis call. Text-embedding retrieval finds the right screenshot
    accurately even for small/dense UI text that CLIP-style image embeddings are
    notoriously bad at reading; sending real pixels (not just the pre-written
    caption) at synthesis time avoids losing visual detail to a lossy summary.
    """
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
