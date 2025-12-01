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
| The Complexity of Automated Cell Type Annotations with GPT-4 | Soumya Luthra et al., 2024 | — | — | Uses GPT-4 for automated cell type annotation; conceptual/benchmark work rather than a standalone FM. |
| BioLLM    | Ping Qiu et al., 2024       | — | — | Large language model for biological / cellular tasks (omics-focused LLM; details in original paper). |
| scGPT-spatial | Chloe Wang et al., 2024 | — | — | Extension of scGPT for spatial transcriptomics; continual pretraining on spatial data. |
| scCello   | Xinyu Yuan et al., 2024     | — | — | scRNA-seq; 23M cross-tissue human cells (CELLxGENE). Rank-based gene ordering; encoder with multi-level pretraining (gene-level MLM, ontology-based cell-type coherence, lineage-aware relational alignment). Fine-tuning: cell type classification; zero-shot cell annotation, marker prediction, novel-cell and cancer drug prediction. |
| scGREAT   | Yuchen Wang et al., 2024    | — | — | Single-cell transformer model (details in original paper). |
| MAMMAL    | Shoshan et al., 2024        | — | — | Trains on bulk/scRNA-seq, amino acids and SMILES; CellXGene Human dataset plus Uniref90, OAS, ZINC, STRING. T5 encoder–decoder; multi-domain SSL (expression masking, protein LM, antibody LM, small-molecule LM, protein interaction LM). Tasks include cell type annotation, drug response, BBB penetration, toxicity, clinical trial prediction, antibody and protein binding. |
| Nicheformer | Anna C. Schaar et al., 2024 | — | — | Transformer model focusing on cellular niche / microenvironment modeling. |
| scmFormer | Jing Xu et al., 2024        | — | — | Transformer-based model for single-cell data (code indicated as searchable). |
| Metric Mirages | Hanchen Wang et al., 2024 | — | — | Explores metric behavior / evaluation in single-cell transformer settings. |
| scEmb     | Kang-Lin Hsieh et al., 2024 | — | — | Model for single-cell embeddings; details in original paper. |
| Cell-ontology guided transcriptome foundation model | Xinyu Yuan et al., 2024 | — | — | Foundation model guided by cell ontology to structure transcriptomic representations. |
| Cell-Graph Compass | Chen Fang et al., 2024 | — | — | Uses graph-based modeling of cell–cell relationships (“cell graph”) with transformer-style components. |
| scGAA     | Tianci Kong et al., 2024    | — | — | Single-cell transformer/graph model; details in original paper. |
| scFusionTTT | Dian Meng et al., 2024    | — | — | Model for fusing single-cell modalities with test-time training ideas. |
| scGenePT  | Ana-Maria Istrate et al., 2024 | — | — | Pretrained transformer for gene-level modeling in single-cell data. |
| ENHANCING GENERATIVE PERTURBATION MODELS WITH LLM-INFORMED GENE EMBEDDINGS | Kaspar Märtens et al., 2024 | — | — | Uses LLM-informed gene embeddings to improve generative perturbation models on single-cell data. |
| scSwinTNet | Huanhuan Dai et al., 2024  | — | — | Swin-transformer-based network for single-cell omics. |
| sclong    | Ding Bai et al., 2024       | — | — | Transformer model for long-range single-cell modeling. |
| WHITE-BOX DIFFUSION TRANSFORMER FOR SINGLE-CELL RNA-SEQ GENERATION | Zhuorui Cui et al., 2024 | — | — | Diffusion transformer for scRNA-seq generation with white-box interpretability focus. |
| A framework for gene representation on spatial transcriptomics | Shenghao Cao et al., 2024 | — | — | Framework to learn gene representations from spatial transcriptomics data. |
| genohoption | Jiabei Cheng et al., 2024 | — | — | Transformer framework for genomics / gene option-style modeling. |
| Cellpatch | Hanwen Zhu et al., 2024     | — | — | Patch-wise transformer model at cell-level (“cell patches”) for single-cell transcriptomics. |
| GRNPT     | Guangzheng Weng et al., 2024 | — | — | Transformer-based model for gene regulatory network (GRN) prediction from single-cell data. |
| Aido.cell | Nicholas Ho et al., 2024    | — | — | Single-cell AI assistant / model; code indicated as searchable. |
| sctel     | Yuanyuan Chen et al., 2024  | — | — | Single-cell transformer model with available code. |
| Toward a privacy-preserving predictive foundation model | Jiayuan Ding et al., 2024 | — | — | Focus on privacy-preserving predictive FM for single-cell / omics. |
| mcBERT    | von Querfurth et al., 2024  | — | — | scRNA-seq; 7M cells from single-tissue human datasets (heart, kidney, PBMC, lung). Cells-as-tokens with value projection; BERT-style encoder. SSL: MLM on cell-level (unmasked cells per patient). Supervised: phenotype classification. |
| CancerFoundation | Theus et al., 2024   | — | — | scRNA-seq; 1M malignant cells from Curated Cancer Cell Atlas. Value binning; attention masking in encoder; iterative MLM with MSE; predicts cell and gene expression. Downstream drug response tasks. |
| Precious3GPT | Galkin et al., 2024      | — | — | Bulk/scRNA-seq, DNA methylation, proteomics and text. Closed-source decoder-only LLaMA-like transformer with modality mappers. Simulates chemical response, cross-omics transfer and clinical conditions; tasks include age prediction and gene classification. |
| LangCell  | Zhao et al., 2024           | — | — | scRNA-seq + natural language. 27M cross-tissue human cells (CELLxGENE). Rank-based gene ordering and text descriptions; dual encoders (cell/text); MLM, contrastive losses and cell-text matching. Tasks: cell type annotation, pathway identification. |
| ScRAT     | Mao et al., 2024            | — | — | scRNA-seq; cells-as-tokens; encoder architecture. Aggregated cell embeddings per sample for phenotype prediction (e.g. disease vs healthy). |
| scPRINT   | Kalfon et al., 2024         | — | — | scRNA-seq; 50M cross-tissue, cross-species cells (CELLxGENE). Uses ESM-2-based gene embeddings; genes sampled and ordered by chromosome. Multi-task pretraining (denoising, bottleneck learning). Supervised pretraining tasks include cell label prediction. |
| scMulan   | Bian et al., 2024           | — | — | scRNA-seq; 10M cross-tissue human cells (hECA). Decoder architecture for conditional cell generation; tasks include cell and metadata annotation. |
| BioFormers | Belgadi & Li et al., 2023  | — | — | scRNA-seq; 8k PBMC cells. Value binning; encoder with MLM (CE). Early small-scale transformer baseline. |
| Geneformer | Theodoris et al., 2023     | — | — | scRNA-seq; 36M cross-tissue human cells (Genecorpus). Rank-based gene ordering; encoder with MLM and gene ID prediction. Tasks: gene function prediction, cell annotation. |
| Universal Cell Embedding | Rosen et al., 2023 | — | — | scRNA-seq; 36M cross-tissue, cross-species cells (CELLxGENE & others). Uses ESM-2-based gene embeddings; sequence formed by gene expression and chromosome positions. Modified MLM with binary CE on gene expression; cell representation from CLS embedding. |
| scGPT     | Cui et al., 2024            | — | — | Multi-omics FM: scRNA-seq, scATAC, CITE-seq, spatial transcriptomics; 33M cross-tissue human non-disease cells (CELLxGENE). Value binning + attention masking; iterative MLM, gene and cell token prediction. Tasks: cell annotation, perturbation, clustering, multimodal embedding, gene function. |
| TOSICA    | Chen et al., 2023           | — | — | scRNA-seq; value projection encoder; primarily for cell type annotation. |
| scMoFormer | Tang et al., 2023          | — | — | scRNA-seq, scATAC, CITE-seq; SVD-based input; encoder plus graph transformers; cross-modality prediction. |
| tGPT      | Shen et al., 2023           | — | — | scRNA-seq; 22M cross-tissue/species, disease and non-disease, organoids. Ordered gene tokens; decoder with next-token prediction and gene ID prediction. |
| SpaFormer | Wen et al., 2023            | — | — | Spatial transcriptomics; cells-as-tokens with value projection; encoder with modified MLM (MSE) for gene expression prediction; tasks: imputation and ST analysis. |
| scFoundation | Hao et al., 2024         | — | — | scRNA-seq; 50M cross-tissue human cells (disease + non-disease) from GEO, SCP, HCA, EMBL-EBI. Two-encoder architecture; modified MLM with MSE for expression prediction. Tasks: drug response, perturbation effect prediction. |
| CellLM    | Zhao et al., 2023           | — | — | scRNA-seq; 1.8M cross-tissue human cells (PanglaoDB, CancerSCEM). Value binning; encoder with contrastive + MLM. Tasks: cancer vs non-disease classification, cell annotation, drug response. |
| scCLIP    | Xiong et al., 2023          | — | — | scRNA-seq + scATAC; 377k fetal human cells. Value projection; encoder with contrastive loss and CE to match modalities. |
| GeneCompass | Yang et al., 2023         | — | — | scRNA-seq; 126M human & mouse cells, disease and non-disease (GEO, SRA, CELLxGENE, etc.). Two-encoder architecture; MLM with CE+MSE for gene ID & expression. Tasks: cell annotation, drug response, gene function. |
| CellPLM   | Wen et al., 2024            | — | — | scRNA-seq + spatial transcriptomics; 11M cross-tissue human cells (HTCA, HCA, GEO). Cells-as-tokens with value projection; encoder with modified MLM (MSE+KL). Tasks: expression imputation, cell annotation, perturbation effect prediction. |
| scMAE     | Kim et al., 2023            | — | — | Single-cell flow cytometry; 6.5M human samples. Concatenated values with learnable protein embeddings; two encoders; MLM with MSE for protein expression; tasks: cell annotation and imputation. |
| CAN/CGRAN | Wang et al., 2023           | — | — | scRNA-seq; value projection encoder; used for cell type annotation. |
| scTranslator | Liu et al., 2023         | — | — | scRNA-seq + CITE-seq; value projection with dual encoders; cross-modality prediction. |
| scTransSort | Jiao et al., 2023         | — | — | scRNA-seq; value projection encoder for cell type annotation. |
| STGRNS    | Xu et al., 2023             | — | — | scRNA-seq; encoder with custom input; focuses on GRN inference. |
| CIForm    | Xu et al., 2023             | — | — | scRNA-seq; value projection encoder; cell type annotation. |
| scFormer  | Cui et al., 2023            | — | — | scRNA-seq; task-specific input; value binning; encoder with modified MLM, cell-token prediction and contrastive loss. Tasks: expression prediction, cell annotation, perturbation. |
| Exceiver  | Connell et al., 2022        | — | — | scRNA-seq; 0.5M cross-tissue human cells (Tabula Sapiens). Value-scaled embeddings; encoder with modified MLM (MSE) for expression prediction; tasks: cell annotation, drug response. |
| TransCluster | Song et al., 2022        | — | — | scRNA-seq; value projection with LDA; encoder used for cell type clustering/annotation. |
| scBERT    | Yang et al., 2022           | — | — | scRNA-seq; 1M cross-tissue human cells (PanglaoDB). Value binning; encoder with MLM (CE) and expression prediction; tasks: cell annotation and unseen cell-type detection. |
| iSEEEK    | Shen et al., 2022           | — | — | scRNA-seq; 11.9M cross-tissue/species cells. Rank-based gene ordering; encoder with MLM (CE). Tasks: marker gene classification and cell-level inference. |
| Multitask learning | Pang et al., 2020  | — | — | scRNA-seq; 160k mouse brain cells (MBA). Value projection; autoencoder with two transformer encoders; modified MLM (MSE) for expression prediction. |

## 4) Multimodal data


| Name      | Paper | GitHub | Hugging Face | Note |
|-----------|-------|--------|:------------:|------|
| IRENE | [A transformer-based representation-learning model with unified processing of multimodal input for clinical diagnostics](https://www.nature.com/articles/s41551-023-01045-x#Abs1) | [RL4M/IRENE](https://github.com/org/repo) | - | the chief complaint, medical images and laboratory test results |
| BioMedCLIP | [A Multimodal Biomedical Foundation Model Trained from Fifteen Million Image–Text Pairs](https://ai.nejm.org/doi/full/10.1056/AIoa2400640) | [microsoft/BiomedCLIP_data_pipeline](https://github.com/microsoft/BiomedCLIP_data_pipeline) | [HF](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) | Pretrained on 15 million biomedical image–text pairs collected from 4.4 million scientific articles | 
| PRISM | [PRISM: A Multi-Modal Generative Foundation Model for Slide-Level Histopathology](https://arxiv.org/pdf/2405.10254) | - | [HF](https://huggingface.co/paige-ai/Prism) | PRISM is a multi-modal generative foundation model for slide-level analysis of H&E-stained histopathology images, has PRISM2|
| CONCH | [A visual-language foundation model for computational pathology](https://www.nature.com/articles/s41591-024-02856-4) | [mahmoodlab/CONCH](https://github.com/mahmoodlab/CONCH)| [HF](https://huggingface.co/MahmoodLab/conch) | CONtrastive learning from Captions for Histopathology (CONCH), a visual-language foundation model developed using diverse sources of histopathology images, biomedical text, and notably over 1.17 million image-caption pairs via task-agnostic pretraining, updated CONCH1.5|
|THEADS| [Molecular-driven Foundation Model for Oncologic Pathology](https://arxiv.org/pdf/2501.16652) | - | - | Pretrained using a multimodal learning approach on a diverse cohort of 47,171 hematoxylin and eosin (H&E)-stained tissue sections, paired with corresponding genomic and transcriptomic profiles | 




## 5) Review paper


| Name      | Paper | GitHub | Hugging Face | Note |
|-----------|-------|--------|:------------:|------|
| GMAI | [Foundation models for generalist medical artificial intelligence]([https://ai.nejm.org/doi/full/10.1056/AIoa2300221](https://www.nature.com/articles/s41586-023-05881-4?fromPaywallRec=false#Abs1)) | - | — | - |
| GAI | [Generative artificial intelligence in medicine](https://www.nature.com/articles/s41591-025-03983-2#Fig1) | - | - | - |
| - |[Overcoming barriers to the wide adoption of single-cell large language models in biomedical research](https://www.nature.com/articles/s41587-025-02846-y#Sec3) | - | - | - |
| - |[Single-cell foundation models: bringing artificial intelligence into cell biology](https://www.nature.com/articles/s12276-025-01547-5) | - | - | - |

## 6) Benchmark paper


| Name      | Paper | GitHub | Hugging Face | Note |
|-----------|-------|--------|:------------:|------|
| Patho-Bench | [Accelerating Data Processing and Benchmarking of AI Models for Pathology](https://arxiv.org/pdf/2502.06750) | [mahmoodlab/Patho-Bench](https://github.com/mahmoodlab/Patho-Bench/tree/main) | [HF](https://huggingface.co/datasets/MahmoodLab/Patho-Bench) | Patho-Bench is designed to evaluate patch and slide encoder foundation models for whole-slide images (WSIs) | 
| PathBench | [PathBench: A comprehensive comparison benchmark for pathology foundation models towards precision oncology](https://arxiv.org/pdf/2505.20202) | [birkhoffkiki/PathBench]([https://github.com/mahmoodlab/Patho-Bench/tree/main](https://github.com/birkhoffkiki/PathBench) | - | PathBench is a comprehensive, multi-task, multi-organ benchmark designed for real-world clinical performance evaluation of pathology foundation models towards precision oncology. This interactive web platform provides standardized evaluation metrics and comparative analysis across 20+ state-of-the-art pathology foundation models. | 


----
Referemces:
[1] https://theislab.github.io/single-cell-transformer-papers/single-cell-transformers
