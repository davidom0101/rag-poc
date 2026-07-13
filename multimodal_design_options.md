# Multimodal Design Options for Screenshot-Heavy Manuals

This document explains the multimodal retrieval/captioning designs considered for this RAG POC, why each was accepted or rejected, and the specific tradeoffs behind the design actually implemented in `app.py`. See `project_spec.md` for the full implementation spec and `Known Issues` for the bugs found along the way.

## The core problem

`gimp-help-2.10-en.pdf` is a screenshot-heavy UI manual. Docling itself extracts **zero** visual content by default — `do_picture_description` and `generate_picture_images` both default to `False`, so a picture node only ever carries a bounding box, never pixels or a description. Any design here has to answer two separate questions:

1. **Retrieval** — how does a screenshot get *found* when a user asks a relevant question?
2. **Synthesis** — what does the LLM actually *see* when answering — the real image, or a proxy for it?

The four options below answer those two questions differently.

---

## Option 1: Pre-Captioning / Text Proxy ("Path B") — implemented first

**Retrieval:** text-embedding similarity, same as body text.
**Synthesis:** the LLM only ever sees a pre-written caption — never the real image.

Docling extracts each screenshot's pixels (`generate_picture_images=True`). Each one is captioned **once**, at ingestion time, via a GPT-4o-mini vision call. The caption becomes a plain `TextNode`, embedded with the same local HuggingFace text embedding model as everything else, and indexed like ordinary prose. At query time, retrieval and synthesis are both 100% text — no image bytes are ever sent again.

**Verified behavior:** worked for retrieval (correctly surfaced the right screenshot's caption for a Toolbox question), but answer quality is capped by the caption's fidelity. Asked to describe the *exact shape* of a specific icon, it answered "likely square or rectangular in shape" — a hedge, because the LLM was reasoning over its own earlier summary of the image, not looking at it again.

**Pros:** cheap, simple, no new dependencies, single vector store.
**Cons:** every answer is bottlenecked by whatever the one caption happened to capture; fine visual detail not mentioned in the caption is permanently lost.

---

## Option 2: CLIP Dual-Embedding Retrieval ("textbook" Path A) — evaluated, rejected

**Retrieval:** two independent embedding-similarity searches — text-vs-text (as usual) **and** text-vs-image via a CLIP joint embedding space — concatenated.
**Synthesis:** real image pixels, sent directly to the LLM.

This is what LlamaIndex's `MultiModalVectorStoreIndex` actually does (confirmed against the installed `llama-index-core` 0.14.23 source, not assumed):

- Requires a genuine `MultiModalEmbedding` — default `"clip:ViT-B/32"`, resolved via `llama-index-embeddings-clip` (**not installed**, ~350MB checkpoint download on first use).
- Text embeddings go to the normal vector store; image embeddings go to a **separate** vector store (`image_vector_store` — a second `MilvusVectorStore` instance would be needed, its own collection, `dim=512` for ViT-B/32).
- `MultiModalVectorIndexRetriever._retrieve()` embeds the query text with *both* the text embed model and the image embed model, searches both stores independently, and concatenates the results. It does **not** work by "text chunk matches → fetch its linked image" — an image only surfaces if CLIP itself judges the query text visually similar to that image's pixels.
- The query engine wired to this (`SimpleMultiModalQueryEngine`) is deprecated as of this llama-index-core version, in favor of `RetrieverQueryEngine.from_args(retriever=..., llm=..., multimodal=True)`.

**Why rejected:** CLIP is trained on general image/concept pairs ("a photo of a cat" ↔ pixels of a cat). It's notoriously bad at reading small, dense text — exactly what GIMP screenshots are full of (menu labels, button captions, dialog text). It would frequently judge the wrong screenshot as the best match for a UI-specific question, and its retrieval mechanics can't be steered around that weakness the way text-embedding retrieval can.

**Pros:** genuine dual-modality retrieval; can find a screenshot even with zero surrounding text.
**Cons:** heavy new dependency, second vector store, and — critically for this document type — the wrong tool for UI-text-dense screenshots.

---

## Option 3: Text Retrieval + Real-Pixel Synthesis ("Path A, revised") — implemented

**Retrieval:** text-embedding similarity only (same single Milvus collection as Option 1 — no CLIP, no second vector store).
**Synthesis:** real image pixels, sent directly to the LLM — not a caption.

This is the hybrid actually in `app.py`. It keeps Option 1's captioning step (still the only way to make screenshots *findable* — most GIMP screenshots have no native caption sentence at all, confirmed 1-of-16 in testing), but the caption's job is now purely to make the image *searchable*, not to be the final answer's evidence:

1. Every screenshot is captioned via GPT-4o-mini vision (as in Option 1) — this is what retrieval matches against.
2. Every screenshot is also saved to disk (`./extracted_images/`), and a ref-map links both the picture's own ref and any native Docling caption ref (e.g. "Figure 3.4 Screenshot of the Toolbox") to that file.
3. `DoclingNodeParser`'s chunk metadata (`doc_items[].self_ref`) is used to tag any *body* text node that mentions a picture or its native caption with the same `linked_image_path` — so native captions become a second, independent path to the same image.
4. At query time (`query_with_images()`), retrieval is ordinary text-embedding similarity. Whatever real image files are linked to the top-k matched nodes are then attached — labeled per-image so the model can tell which screenshot is which — to the GPT-4o-mini vision call alongside the retrieved text.

**Why chosen:** avoids CLIP's UI-text blindness (retrieval is driven by accurate text embeddings, not image embeddings) while still answering from real pixels instead of a lossy summary.

**Verified, honestly:** the retrieval/linking mechanics are confirmed correct — directly inspected retrieved nodes' metadata after a fresh index reconnect and confirmed the right image resolves. Raising Docling's `images_scale` from the default `1.0` to `3.0` (84×172px → 253×515px for the Toolbox screenshot) meaningfully improved crop quality. But re-running the *identical* icon-shape question three times against the *same, correctly-attached* image produced three different answers — one correct ("rectangular selection tool"), two wrong ("pen or drawing tool", "scissors"). **The pipeline's engineering is verified correct; fine-grained icon-shape accuracy is bounded by GPT-4o-mini vision's own reliability on small icon-grid crops, not by anything further fixable here.** See `project_spec.md` Known Issue 5.

**Pros:** no new dependencies, one vector store, retrieval accuracy suited to UI-text-heavy screenshots, real-pixel-grounded synthesis.
**Cons:** still requires the captioning pass at ingestion (one vision API call per screenshot); doesn't help with a screenshot that has *neither* a native caption *nor* gets a distinctive generated caption; final answer accuracy on small fine-grained visual details is still limited by the vision model itself, not the retrieval design.

---

## Option 4: Docling's Built-in Local VLM Captioning — noted, not used

Docling has its own opt-in picture-description step (`PdfPipelineOptions.do_picture_description=True`), which defaults to running a local `SmolVLM-256M-Instruct` model (via `AutoInlineVlmEngineOptions`) to caption each picture during conversion itself, rather than calling out to an external API afterward.

**Why not used:** the rest of this stack already runs entirely on OpenAI's API (embeddings are the one local piece — `BAAI/bge-small-en-v1.5` via HuggingFace). Adding a second, different local VLM just for picture captions would mean maintaining two captioning code paths instead of one, and a 256M-parameter local model is meaningfully weaker than GPT-4o-mini vision for reading dense UI text. If a fully-offline/no-API-cost pipeline were ever needed, this is the natural place to look.

---

## Comparison at a glance

| | Retrieval mechanism | Synthesis sees | New dependency | Verified weakness |
|---|---|---|---|---|
| 1. Text Proxy | Text embeddings | Pre-written caption | None | Answers bounded by caption fidelity |
| 2. CLIP Dual-Embedding | Text embeddings **+** CLIP image embeddings | Real pixels | `llama-index-embeddings-clip` (~350MB) | Poor at dense/small UI text; wrong-screenshot risk |
| 3. Text + Real Pixels (chosen) | Text embeddings only | Real pixels | None | Vision-model reliability on fine icon detail, not retrieval |
| 4. Docling Local VLM | (captioning only, not a retrieval design) | N/A | Local SmolVLM-256M | Weaker than GPT-4o-mini at reading UI text |
