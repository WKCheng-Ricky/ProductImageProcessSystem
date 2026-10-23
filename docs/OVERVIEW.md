# Product Image Processing System - Strategic Overview

Table of Content:

- [Product Image Processing System - Strategic Overview](#product-image-processing-system---strategic-overview)
  - [Business Challenge](#business-challenge)
  - [Core Architectural Decision](#core-architectural-decision)
  - [Processing Pipeline](#processing-pipeline)
    - [Search Capabilities](#search-capabilities)
    - [Product Grouping](#product-grouping)
  - [Key Technologies](#key-technologies)
    - [CLIP (Contrastive Language-Image Pre-training)](#clip-contrastive-language-image-pre-training)
    - [U2Net (U-square Network)](#u2net-u-square-network)
    - [LAION Aesthetic Predictor](#laion-aesthetic-predictor)
    - [Milvus Vector Database](#milvus-vector-database)
  - [Aesthetic Scoring Approach](#aesthetic-scoring-approach)
    - [1. Clarity](#1-clarity)
    - [2. Composition](#2-composition)
    - [3. Defects](#3-defects)
  - [Operational Workflow](#operational-workflow)
    - [1. Image Ingestion](#1-image-ingestion)
    - [2. Automated Cleaning](#2-automated-cleaning)
    - [3. Quality Assessment](#3-quality-assessment)
    - [4. Embedding Generation](#4-embedding-generation)
    - [5. Similarity Search \& Grouping](#5-similarity-search--grouping)
    - [6. Confidence-Based Review](#6-confidence-based-review)
  - [Technology Stack Summary](#technology-stack-summary)
  - [Scalability Considerations](#scalability-considerations)
  - [Business Value](#business-value)
  - [Possible improvement](#possible-improvement)
  - [References \& Sources](#references--sources)
    - [CLIP (Contrastive Language-Image Pre-training)](#clip-contrastive-language-image-pre-training-1)
    - [U2Net (Salient Object Detection)](#u2net-salient-object-detection)
    - [LAION Aesthetic Predictor](#laion-aesthetic-predictor-1)
    - [Milvus Vector Database](#milvus-vector-database-1)
    - [Other](#other)

<div style="page-break-after: always;"></div>
   
## Business Challenge

Managing 2M+ SKUs with inconsistent image quality across multiple sources:
- **Quality Issues**: Varying backgrounds, lighting, angles, and resolutions
- **Watermarks & Text**: Promotional overlays obscuring products
- **Composite Images**: Multiple products in single collage photos
- **Duplicate Detection**: Same products appearing under different SKU codes
- **Main Image Selection**: Choosing the best representative photo from multiple options

**Goal**: Automatically process, clean, score, and organize product images at scale to improve catalog quality and enable intelligent product matching.

---

## Core Architectural Decision

**Strategy: Preprocess Images BEFORE Generating Embeddings**

```
Raw Image → Clean & Standardize → Generate Embedding → Search & Match
```

**Rationale**: Background clutter, watermarks, and text overlays severely distort embeddings. A shampoo bottle photographed on a white background vs. a kitchen counter produces dramatically different vector representations, preventing accurate similarity matching.

By cleaning images first, we ensure embeddings capture actual product features rather than environmental noise.

![](./screenshoot/background_removal.png)

---

<div style="page-break-after: always;"></div>

## Processing Pipeline

![flowdiagram](./screenshoot/overview_processiong_pipeline.png)

<div style="page-break-after: always;"></div>

### Search Capabilities

**Path 1: Image → Image Search**
- Upload a product image, find visually similar products in catalog
- Use case: "Show me products that look like this competitor item"

![](./screenshoot/Image_search.png)

<div style="page-break-after: always;"></div>

**Path 2: Text → Image Search**
- Enter text description, find matching product images
- Use case: "Find all blue wireless headphones"
- 
![](./screenshoot/Text%20Search.png)

```
  How CLIP Search Works                                                           
                                                                                  
  STORING:                                                  
  Image → CLIP Image Encoder → vector [512 dims] → Milvus                         
                                                                                  
  SEARCHING:                                                      
  Text → CLIP Text Encoder → vector [512 dims] → Search Milvus                    
                                                      ↓                           
                                                Finds similar IMAGE vectors       
```

### Product Grouping

**How It Works**:
- Embeddings with >90% similarity are clustered as same product
- DBSCAN algorithm groups products without needing predefined categories
- Enables automatic duplicate detection across different SKU codes

<div style="page-break-after: always;"></div>

**Business Rules**:
| Similarity | Action |
|------------|--------|
| > 90% | Auto-match (same product) |
| 80-90% | Similar product (variant/related) |
| < 80% | Different product |

---

## Key Technologies

### CLIP (Contrastive Language-Image Pre-training)

**CLIP** is a neural network developed by OpenAI<sup>[[1]](#ref1)</sup> that understands images through natural language. Trained on 400M+ image-text pairs<sup>[[2]](#ref2)</sup> from the internet, CLIP learns to connect visual concepts with human descriptions.

**How It Works**:
- Uses two neural networks<sup>[[2]](#ref2)</sup>: one for images (Vision Transformer), one for text
- Projects both images and text into the same mathematical space (embeddings)
- Images with similar visual characteristics cluster together in this space
- Enables "semantic search" - finding visually similar products

**Why We Chose It**:
1. **Semantic Understanding**: Captures high-level product attributes (style, color, category) beyond pixel matching
2. **No Training Required**: Pre-trained on massive datasets, works immediately on our product catalog
3. **Robust Vectors**: 512-1024 dimensional embeddings that mathematically represent product appearance
4. **Dual Purpose**: Same technology powers both similarity search and aesthetic quality scoring
5. **Production-Ready**: Industry standard for visual search (Pinterest's Canvas AI<sup>[[3]](#ref3)</sup>, e-commerce platforms)

**Implementation**:
- Multiple model sizes available (ViT-B-32, ViT-L-14, ViT-H-14) to balance speed vs. accuracy
- Embeddings stored in vector database for fast similarity queries
- Fine-tuned for aesthetic scoring using prompt engineering

<div style="page-break-after: always;"></div>

### U2Net (U-square Network)

U<sup>2<sup>Net is a specialized deep learning model for salient object detection<sup>[[4]](#ref4)</sup> - identifying and precisely segmenting the main subject in an image while removing backgrounds.

**How It Works**:
- Two-level nested architecture<sup>[[4]](#ref4)</sup> (U-shaped network within U-shaped network)
- Analyzes images at multiple scales simultaneously
- Outputs pixel-perfect masks separating product from background
- Handles complex cases: transparent objects, fine details, irregular shapes

**Why We Chose It**:
1. **Precision**: Accurately segments products with complex boundaries (glass bottles, fabric, jewelry)
2. **Zero Configuration**: Pre-trained model works out-of-the-box<sup>[[5]](#ref5)</sup>, no manual labeling needed
3. **Lightweight & Fast**: U2NetP variant has only 1.13M parameters (4.7 MB)<sup>[[6]](#ref6)</sup>, enabling efficient batch processing
4. **Embedding Quality**: Removing backgrounds is critical for consistent similarity matching

**Impact**: Without background removal, embedding quality drops significantly. Products photographed in different environments (white studio vs. retail shelf vs. home setting) appear "different" to the AI, breaking similarity search.


### LAION Aesthetic Predictor

LAION is a neural network trained on human aesthetic preferences<sup>[[7]](#ref7)</sup> to score image quality on a 1-10 scale. Developed by LAION (Large-scale Artificial Intelligence Open Network)<sup>[[8]](#ref8)</sup> as part of their open dataset initiatives.

**How It Works**:
- Built on top of CLIP embeddings
- Trained on millions of human-rated images from various sources
- Learns correlation between visual features and perceived quality
- Outputs a single aesthetic score predicting human preference

**Why We Chose It**:
1. **Human-Aligned Scoring**: Trained on actual human ratings, not arbitrary metrics
2. **CLIP Integration**: Uses same embedding model we already have, minimal overhead
3. **Production Proven**: Used to filter LAION-5B dataset<sup>[[9]](#ref9)</sup>, processing billions of images
4. **No Custom Training**: Works immediately on product images without fine-tuning

**Our Application**:
- Score raw images before processing (baseline quality)
- Compare multiple product photos to select "main image"
- Flag low-scoring images for manual review or re-shooting
- Quality threshold (default 0.3) filters out unusable images

### Milvus Vector Database

**What It Is**: Specialized database for storing and searching high-dimensional vectors<sup>[[10]](#ref10)</sup> (embeddings).

**Why We Need It**:
- Traditional databases can't efficiently search 512-dimensional vectors across millions of records
- Enables sub-second similarity queries<sup>[[11]](#ref11)</sup>: "find products that look like this one"
- Supports clustering algorithms for duplicate detection
- Scalable to billions of embeddings<sup>[[12]](#ref12)</sup>

---

## Aesthetic Scoring Approach

Our AI-powered quality scorer evaluates images across three critical dimensions:

### 1. Clarity
- Sharp focus vs. blurry
- Proper lighting vs. dark/overexposed
- Detail visibility

### 2. Composition
- Product framing and centering
- Professional angles
- Appropriate zoom level

### 3. Defects
- Watermark presence
- Text overlay detection
- Visual obstructions

**Scoring Method**: Using CLIP's language understanding, we define "positive" and "negative" text prompts for each dimension:

**Clarity Examples**:
- Positive: "sharp in-focus product", "well-lit product photo"
- Negative: "blurry out-of-focus product", "dark poorly lit photo"

The model compares images against these descriptions to generate 0-10 quality scores, enabling automated "best image" selection when multiple photos exist per SKU.

---

<div style="page-break-after: always;"></div>

## Operational Workflow

### 1. Image Ingestion
- Accept raw product photos from any source (suppliers, web scraping, internal photography)
- Handle various formats, sizes, and quality levels

### 2. Automated Cleaning
- Detect and split collage images into individual products
- Remove backgrounds using U2Net
- Detect and remove watermarks/text overlays
- Standardize to consistent size and format

### 3. Quality Assessment
- Score each image on 0-10 scale
- Flag low-quality images for review or re-shooting
- Select best representative photo when multiple options exist

### 4. Embedding Generation
- Convert cleaned images to CLIP vectors
- Store in Milvus vector database with associated metadata

### 5. Similarity Search & Grouping
- Enable "find similar products" functionality
- Detect potential duplicate SKUs (same product, different codes)
- Cluster related product variations

### 6. Confidence-Based Review
- **Similarity > 90%**: Auto-match with high confidence
- **Similarity < 90%**: Queue for human review

---

<div style="page-break-after: always;"></div>

## Technology Stack Summary

| Component | Technology | Reasoning |
|-----------|------------|-----------|
| Embedding Model | CLIP (OpenAI) | Industry standard, no training required |
| Aesthetic Scoring | LAION | Human-aligned quality scoring, no custom training needed |
| Background Removal | U2Net | Best accuracy for product segmentation |
| Vector Database | Milvus | Scalable, production-grade vector search |
| Text Detection | EasyOCR | Open-source, multi-language support |
| ML Framework | PyTorch | CLIP and U2Net native compatibility |
| Clustering | DBSCAN | Density-based, no predefined cluster count needed |
| Image Processing | OpenCV (cv2) | Standard library for crop, resize, transformation operations |

---

## Scalability Considerations

**Current Scope**: 2M+ SKUs, multiple images each (6M-10M+ total images)

**Computational Requirements**:
- **Background Removal**: 0.5-2s per image (CPU), faster with GPU
- **Embedding Generation**: ~0.1s per image (GPU recommended for batch processing)
- **Vector Search**: Sub-second queries across millions of embeddings

**Deployment Options**:
- **Development**: Milvus Lite (embedded database)
- **Production**: Milvus cluster (distributed, horizontally scalable)

**Estimated Processing Time** (single CPU):
- 10M images × 2s average = ~231 days continuous processing
- **Recommendation**: GPU acceleration + parallel processing for initial catalog ingestion

---

## Business Value

1. **Catalog Quality**: Automated cleaning improves customer-facing image consistency
2. **Duplicate Detection**: Identify redundant SKUs, consolidate inventory
3. **Search Enhancement**: Enable visual product search ("find products like this")
4. **Cost Reduction**: Reduce manual image editing and SKU consolidation effort
5. **Scalability**: Handle catalog growth without proportional labor increase

## Possible improvement

There is some significant shortcut taken when developing this prototype / demostration code.

- The current code base only use `CPU`, as it is easier to run on GCP. 
- Machine learning model with `U2Net`, `CLIP` and `LAION` could be significantly sped up with `GPU`
- Model selection is limited to weaker for faster processing speed, scarficing accuracy. 
- The default model, `ViT-B-32` (chosen for fast deployment; `ViT-L-14` is optional), is weaker than larger CLIP models: `ViT-L-14` reaches only 75.3 zero-shot top-1 accuracy on ImageNet-1k<sup>[[13]](#ref13)</sup>.
- For rapid developement, python is used for rapid development rate, enormous documentation, and ease of managing package, however:
- Web Hosting, Model processing and Database connection all using Python, which may not be ideal.
- Python excel at handling Machine learning model, but it can have even less overhead with C++ or Rust
- Python may not be the best server language, Java, C# maybe a better choice.
- Python will have trouble hanlding large dataset, namely, the native python loop performance is slow.

<div style="page-break-after: always;"></div>

## References & Sources

### CLIP (Contrastive Language-Image Pre-training)
- <a name="ref1"></a>[1] [CLIP: Connecting text and images | OpenAI](https://openai.com/index/clip/) - Official announcement and overview
- <a name="ref2"></a>[2] [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/pdf/2103.00020) - Full paper with architectural details including dual encoder design
- <a name="ref3"></a>[3] [Pinterest debuts Canvas AI at VB Transform](https://venturebeat.com/ai/pinterest-debuts-canvas-ai-at-vb-transform-reshaping-visual-discovery) - Production use of CLIP in e-commerce

### U2Net (Salient Object Detection)
- <a name="ref4"></a>[4] [U²-Net: Going Deeper with Nested U-Structure (arXiv)](https://arxiv.org/abs/2005.09007) - Original research paper
- <a name="ref5"></a>[5] [GitHub - xuebinqin/U-2-Net](https://github.com/xuebinqin/U-2-Net) - Official repository with model specifications
- <a name="ref6"></a>[6] [U2-Net: An Approach to Efficient Background Removal | LearnOpenCV](https://learnopencv.com/u2-net-image-segmentation/) - Technical guide and performance metrics

### LAION Aesthetic Predictor
- <a name="ref7"></a>[7] [LAION Aesthetics | LAION](https://laion.ai/blog/laion-aesthetics/) - Official blog post on aesthetic predictor
- <a name="ref8"></a>[8] [GitHub - LAION-AI/aesthetic-predictor](https://github.com/LAION-AI/aesthetic-predictor) - Official repository with model and usage
- <a name="ref9"></a>[9] [LAION-5B: An open large-scale dataset for training next generation image-text models](https://laion.ai/blog/laion-5b/) - Dataset documentation showing aesthetic filtering at scale

### Milvus Vector Database
- <a name="ref10"></a>[10] [Milvus | High-Performance Vector Database Built for Scale](https://milvus.io/) - Official documentation
- <a name="ref11"></a>[11] [Scalable and Blazing Fast Similarity Search with Milvus](https://milvus.io/blog/scalable-and-blazing-fast-similarity-search-with-milvus-vector-database.md) - Performance benchmarks
- <a name="ref12"></a>[12] [Scaling Search with Milvus: Handling Massive Datasets](https://zilliz.com/blog/scale-search-with-milvus-handle-massive-datasets-with-ease) - Billion-scale deployment case studies

### Other
- <a name="ref13"></a>[13] [Hugging Face - Model Card for CLIP ViT-L/14 - LAION-2B](https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K) - Model card from Hugging Face