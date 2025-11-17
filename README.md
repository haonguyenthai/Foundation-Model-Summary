# Foundation-Model-Summary

Summaries of foundation models (FMs) across modalities.

## 1) Clinical Data


| Name      | Paper | GitHub | Hugging Face | Note |
|-----------|-------|--------|:------------:|------|
| Foresight | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2589750024000256?pes=vor&utm_source=chemport&getft_integrator=chemport) | [CogStack/Foresight](https://github.com/CogStack/Foresight/tree/main) | — | EHR-focused FM; concept extraction & phenotyping. |




## 2) Image Data


| Name      | Paper | GitHub | Model weight | Note |
|-----------|-------|--------|:------------:|------|
| Vision FM | [VisionFM: A Vision Foundation Model for Generalist Ophthalmic Artificial Intelligence](https://ai.nejm.org/doi/full/10.1056/AIoa2300221) | [ABILab-CUHK/VisionFM](https://github.com/ABILab-CUHK/VisionFM/tree/main) | — | eight common ophthalmic imaging modalities including fundus photography, optical coherence tomography (OCT), fundus fluorescein angiography (FFA), slit lamp, B-scan ultrasound, external eye imaging, MRI, and ultrasound biomicroscopy (UBM). |
| RoentGen | [A vision–language foundation model for the generation of realistic chest X-ray images](https://www.nature.com/articles/s41551-024-01246-y#Abs1) | [StanfordMIMI/RoentGen](https://github.com/StanfordMIMI/RoentGen) | - | A domain-adapted latent diffusion model capable of generating high-quality, text-conditioned chest X-rays (CXRs). |
| CHIEF | [A pathology foundation model for cancer diagnosis and prognosis prediction](https://www.nature.com/articles/s41586-024-07894-z) | [hms-dbmi/CHIEF](https://github.com/hms-dbmi/CHIEF) | [Docker](https://hub.docker.com/r/chiefcontainer/chief) | CHIEF using 60,530 whole-slide mimages (WSIs) spanning 19 distinct anatomical sites. |
| Prov-Gigapath | [A whole-slide foundation model for digital pathology from real-world data](https://www.nature.com/articles/s41586-024-07441-w) | [pro-gigapath/prov-gigapath](https://github.com/prov-gigapath/prov-gigapath) | [HF](https://huggingface.co/prov-gigapath/prov-gigapath) | A whole-slide pathology foundation model pretrained on 1.3 billion 256 × 256 pathology image tiles in 171,189 whole slides from more than 30,000 patients covering 31 major tissue types  |
| BEPH | [A foundation model for generalizable cancer diagnosis and survival prediction from histopathological images](https://www.nature.com/articles/s41467-025-57587-y) | [Zhcyoung/BEPH](https://github.com/Zhcyoung/BEPH) | [Drive](https://drive.google.com/file/d/19Fu3dw3G4i2gPXijzrxfaQ2D_xcqNdNz/view?usp=sharing) | BEPH is pre-trained on 11 million histopathological images from TCGA with self-supervised learning | 
| AIM | [Foundation model for cancer imaging biomarkers](https://www.nature.com/articles/s42256-024-00807-9#Abs1) | [AIM-Harvard/foundation-cancer-image-biomarker](https://github.com/AIM-Harvard/foundation-cancer-image-biomarker/tree/master) | - | A foundation model for cancer imaging biomarker discovery by training a convolutional encoder through self-supervised learning using a comprehensive dataset of 11,467 radiographic lesions |
| USFM | [USFM: A universal ultrasound foundation model generalized to tasks and organs towards label efficient image analysis](https://www.sciencedirect.com/science/article/pii/S1361841524001270) | [openmedlab/USFM](https://github.com/openmedlab/USFM) | [Drive](https://drive.google.com/file/d/1KRwXZgYterH895Z8EpXpR1L1eSMMJo4q/view?usp=sharing) | A large-scale Multi-organ, Multi-center, and Multi-device US database was built, comprehensively containing over two million US images |
| UNI | [Towards a general-purpose foundation model for computational pathology](https://www.nature.com/articles/s41591-024-02857-3) | [mahmoodlab/UNI](https://github.com/mahmoodlab/UNI) | [HF](https://huggingface.co/MahmoodLab/uni) | ViT-l/16, UNI2-h: ViT-h/14-reg8 | 
| Virchow2 | [VIRCHOW2: SCALING SELF-SUPERVISED MIXEDMAGNIFICATION MODELS IN PATHOLOGY](https://arxiv.org/pdf/2408.00738) | [HF](https://huggingface.co/paige-ai/Virchow2) | - | Virchow2, a 632 million parameter vision transformer, Virchow2G, a 1.9 billion parameter vision transformer, and Virchow2G Mini, a 22 million parameter distillation of Virchow2G, each trained with 3.1 million histopathology whole slide images, with diverse tissues, originating institutions, and stains | 





## 3) Genomic data


| Name      | Paper | GitHub | Hugging Face | Note |
|-----------|-------|--------|:------------:|------|
| SToFM | [SToFM: a Multi-scale Foundation Model for Spatial Transcriptomics](https://paper.link](https://arxiv.org/pdf/2507.11588) | [PharMolix/SToFM](https://github.com/PharMolix/SToFM/tree/main) | [HF]([https://huggingface.co/org/model](https://drive.google.com/drive/folders/1mHE8gf8MAPwzZoEB0vwOOfQ4lz3H_-xo?usp=sharing)) | A multi-scale Spatial Transcriptomics Foundation Model. SToFM first performs multi-scale information extraction on each ST slice, to construct a set of ST sub-slices that aggregate macro-, micro- and gene-scale information. Then an SE(2) Transformer is used to obtain high-quality cell representations from the sub-slices. |
| CellPLM | [CellPLM: Pre-training of Cell Language Model Beyond Single Cells](https://www.biorxiv.org/content/10.1101/2023.10.03.560734v1.full.pdf) | [OmicsML/CellPLM](https://github.com/OmicsML/CellPLM) | [Dropbox](https://www.dropbox.com/scl/fo/i5rmxgtqzg7iykt2e9uqm/h?rlkey=o8hi0xads9ol07o48jdityzv1&dl=0) | CellPLM is the first single-Cell Pre-trained Language Model that encodes cell-cell relations and it consistently outperforms existing pre-trained and non-pre-trained models in diverse downstream tasks, with 100x higher inference speed compared to existing pre-trained models|
| KRONOS | [A Foundation Model for Spatial Proteomics](https://arxiv.org/pdf/2506.03373) | [mahmoodlab/KRONOS](https://github.com/mahmoodlab/KRONOS)| [HF](https://huggingface.co/MahmoodLab/KRONOS) | KRONOS is a panel-agnostic foundation model for spatial proteomics, self-supervised on 47 million single-marker patches spanning 175 protein markers, 16 tissue types, 8 imaging platforms and 5 institutions | 
| scConcept | [scConcept: Contrastive pretraining for technology-agnostic single-cell representations beyond reconstruction](https://www.biorxiv.org/content/biorxiv/early/2025/10/15/2025.10.14.682419.full.pdf) | [theislab/scConcept](https://github.com/theislab/scConcept) | - | Defines a “cell view” as any arbitrary subset of genes from a cell, Samples two disjoint gene panels per mini-batch and creates two views of every cell (view A, view B) |
| scFoundation | [Large-scale foundation model on single-cell transcriptomics](https://www.nature.com/articles/s41592-024-02305-7) | [biomap-research/scFoundation](https://github.com/biomap-research/scFoundation) | - | A large-scale pretrained model scFoundation with 100M parameters. scFoundation was based on the xTrimoGene architecture and trained on over 50 million human single-cell transcriptomics data, which contain high-throughput observations on the complex molecular features in all known types of cells. scFoundation is a large-scale model in terms of the size of trainable parameters, dimensionality of genes and the number of cells used in the pre-training |

## 3) Multimodal data


| Name      | Paper | GitHub | Hugging Face | Note |
|-----------|-------|--------|:------------:|------|
| IRENE | [A transformer-based representation-learning model with unified processing of multimodal input for clinical diagnostics](https://www.nature.com/articles/s41551-023-01045-x#Abs1) | [RL4M/IRENE](https://github.com/org/repo) | - | the chief complaint, medical images and laboratory test results |
| BioMedCLIP | [A Multimodal Biomedical Foundation Model Trained from Fifteen Million Image–Text Pairs](https://ai.nejm.org/doi/full/10.1056/AIoa2400640) | [microsoft/BiomedCLIP_data_pipeline](https://github.com/microsoft/BiomedCLIP_data_pipeline) | [HF](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) | Pretrained on 15 million biomedical image–text pairs collected from 4.4 million scientific articles | 
| PRISM | [PRISM: A Multi-Modal Generative Foundation Model for Slide-Level Histopathology](https://arxiv.org/pdf/2405.10254) | - | [HF](https://huggingface.co/paige-ai/Prism) | PRISM is a multi-modal generative foundation model for slide-level analysis of H&E-stained histopathology images, has PRISM2|
| CONCH | [A visual-language foundation model for computational pathology](https://www.nature.com/articles/s41591-024-02856-4) | [mahmoodlab/CONCH](https://github.com/mahmoodlab/CONCH)| [HF](https://huggingface.co/MahmoodLab/conch) | CONtrastive learning from Captions for Histopathology (CONCH), a visual-language foundation model developed using diverse sources of histopathology images, biomedical text, and notably over 1.17 million image-caption pairs via task-agnostic pretraining, updated CONCH1.5|
|THEADS| [Molecular-driven Foundation Model for Oncologic Pathology](https://arxiv.org/pdf/2501.16652) | - | - | Pretrained using a multimodal learning approach on a diverse cohort of 47,171 hematoxylin and eosin (H&E)-stained tissue sections, paired with corresponding genomic and transcriptomic profiles | 




## 3) Review paper


| Name      | Paper | GitHub | Hugging Face | Note |
|-----------|-------|--------|:------------:|------|
| GMAI | [Foundation models for generalist medical artificial intelligence]([https://ai.nejm.org/doi/full/10.1056/AIoa2300221](https://www.nature.com/articles/s41586-023-05881-4?fromPaywallRec=false#Abs1)) | - | — | - |
| GAI | [Generative artificial intelligence in medicine](https://www.nature.com/articles/s41591-025-03983-2#Fig1) | - | - | - |
| - |[Overcoming barriers to the wide adoption of single-cell large language models in biomedical research](https://www.nature.com/articles/s41587-025-02846-y#Sec3) | - | - | - |
| - |[Single-cell foundation models: bringing artificial intelligence into cell biology](https://www.nature.com/articles/s12276-025-01547-5) | - | - | - |

## 3) Benchmark paper


| Name      | Paper | GitHub | Hugging Face | Note |
|-----------|-------|--------|:------------:|------|
| Patho-Bench | [Accelerating Data Processing and Benchmarking of AI Models for Pathology](https://arxiv.org/pdf/2502.06750) | [mahmoodlab/Patho-Bench](https://github.com/mahmoodlab/Patho-Bench/tree/main) | [HF](https://huggingface.co/datasets/MahmoodLab/Patho-Bench) | Patho-Bench is designed to evaluate patch and slide encoder foundation models for whole-slide images (WSIs) | 

