RDM - Rivet Drawing Matcher
Overview

RDM (Rivet Drawing Matcher) is an experimental industrial image retrieval system designed to identify the most likely engineering drawing corresponding to a photographed rivet or mechanical component.

In manufacturing and maintenance scenarios, operators may encounter damaged, corroded, or unidentified parts whose drawing numbers are unavailable. Locating the correct engineering drawing often requires manually searching through large drawing repositories, which is time-consuming and heavily dependent on domain expertise.

This project explores whether modern visual representation learning techniques can bridge the gap between engineering drawings and real-world component photographs, enabling similarity-based drawing retrieval.

Rather than focusing solely on model accuracy, the project investigates the practical challenges of cross-domain retrieval in industrial environments.

Project Objectives

The primary goals of this project are:

Build an engineering drawing retrieval pipeline.
Evaluate self-supervised visual features for industrial search tasks.
Explore metric learning methods for domain-specific retrieval.
Analyze the impact of drawing quality on retrieval performance.
Investigate the challenges of matching real-world photographs to engineering drawings.
Application Scenario

A typical workflow is:

Unknown Rivet Photo
         │
         ▼
 Feature Extraction
         │
         ▼
 Similarity Search
         │
         ▼
 Candidate Drawings

The system accepts a photograph of a rivet or mechanical component and returns the most visually similar engineering drawings from a local drawing repository.

Potential applications include:

Spare part identification
Rivet drawing lookup
Legacy drawing management
Workshop maintenance support
Manufacturing documentation retrieval
System Architecture
                    Query Image
                          │
                          ▼
                 Feature Extraction
                          │
          ┌───────────────┴───────────────┐
          │                               │
          ▼                               ▼
    DINOv2 Encoder                 ArcFace Encoder
          │                               │
          └───────────────┬───────────────┘
                          ▼
                   Embedding Vector
                          │
                          ▼
                   Vector Database
                          │
                          ▼
                  Similarity Retrieval
                          │
                          ▼
                   Top-K Candidates

The project supports two retrieval approaches.

Retrieval Methods
DINOv2 Retrieval

A zero-shot retrieval pipeline based on self-supervised vision transformers.

Characteristics:

No training required
Fast deployment
Strong feature generalization
Suitable for rapid experimentation

DINOv2 embeddings are extracted from engineering drawings and stored in a vector index. Query photographs are converted into the same embedding space and compared using cosine similarity.

ArcFace Retrieval

A metric-learning based retrieval pipeline.

The approach generates synthetic variants of engineering drawings and trains a supervised embedding model using ArcFace loss.

Characteristics:

Domain-specific feature learning
Improved class separation
Better control over embedding space

The trained model is then used to build a retrieval index for similarity search.

Data Processing Pipeline

To improve robustness against industrial image variations, the project includes several preprocessing and augmentation stages.

Drawing Recropping

Engineering drawings often contain:

Large borders
Title blocks
Revision tables
Empty margins

The recropping pipeline attempts to isolate the main component region and remove irrelevant content.

Data Augmentation

Synthetic image generation is used to reduce the visual gap between engineering drawings and real-world photographs.

Augmentations include:

Rotation
Scaling
Perspective transformation
Blur
Noise injection
Brightness variation
Texture overlays
Scratch simulation
Rust-like color distortion
Random erasing
Technology Stack
Deep Learning
PyTorch
TorchVision
DINOv2
ArcFace
Computer Vision
OpenCV
Pillow
Retrieval
FAISS
Cosine Similarity Search
Utilities
NumPy
tqdm
Project Structure
RDM/
│
├── inference.py
├── train.py
├── augment.py
├── recrop.py
├── utils.py
│
├── data/
├── augmented/
├── labels/
├── models/
│
├── requirements.txt
└── README.md
Key Technical Challenges

One of the primary goals of this project was to investigate why industrial retrieval systems often perform significantly worse than expected despite using modern deep learning models.

During experimentation, the retrieval performance did not reach production-level accuracy.

Analysis showed that the dominant limitation was data quality rather than model architecture.

Noisy Engineering Drawings

The drawing repository contained several issues:

Large borders occupied substantial image area
Text annotations dominated visual content
Some drawings contained little or no actual component geometry
Drawing styles were inconsistent
Certain files were effectively invalid samples

As a result, feature extractors frequently focused on text regions, tables, and layout structures instead of the component itself.

Cross-Domain Gap

A significant visual gap existed between:

Engineering Drawings
Clean
High contrast
Structured
CAD-like appearance
Real Component Photographs
Rusted
Damaged
Blurred
Occluded
Complex backgrounds

This domain discrepancy proved to be a major challenge for retrieval accuracy.

Model Limitations

Although DINOv2 and ArcFace provided meaningful feature representations, they could not fully compensate for:

Poor-quality source drawings
Missing component information
Domain mismatch
Limited dataset scale

The experiments suggested that improving dataset quality would likely yield larger gains than increasing model complexity.

Experimental Findings

Several observations emerged during development:

DINOv2 provided surprisingly strong baseline retrieval performance without any training.
ArcFace improved feature discrimination but remained constrained by dataset quality.
Drawing preprocessing had a larger impact than expected.
Retrieval quality depended heavily on the visibility of component structure.
Data quality became the primary bottleneck before model architecture.
Lessons Learned

This project reinforced an important practical lesson in industrial computer vision:

Data quality often matters more than model complexity.

While modern architectures such as Vision Transformers and metric-learning models are powerful, their effectiveness is fundamentally limited by the quality and consistency of the underlying data.

Future improvements would focus on:

Automatic removal of invalid drawings
Text and border suppression
Better component segmentation
Domain adaptation techniques
Larger industrial datasets
Multi-modal retrieval approaches
Future Work

Potential future directions include:

Vision-language retrieval models
Contrastive learning frameworks
Industrial-domain pretraining
OCR-aware drawing filtering
Hybrid image + metadata retrieval
Human-in-the-loop retrieval refinement
Repository Status

This project should be considered an experimental proof-of-concept rather than a production-ready system.

Its primary value lies in exploring industrial cross-domain retrieval challenges and understanding the practical limitations imposed by real-world engineering data.
