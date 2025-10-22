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





## 3) Genomic data


| Name      | Paper | GitHub | Hugging Face | Note |
|-----------|-------|--------|:------------:|------|
| ModelName | [Paper](https://paper.link) | [org/repo](https://github.com/org/repo) | [Model Card](https://huggingface.co/org/model) | One-line summary or scope. |

## 3) Multimodal data


| Name      | Paper | GitHub | Hugging Face | Note |
|-----------|-------|--------|:------------:|------|
| IRENE | [A transformer-based representation-learning model with unified processing of multimodal input for clinical diagnostics](https://www.nature.com/articles/s41551-023-01045-x#Abs1) | [RL4M/IRENE](https://github.com/org/repo) | - | the chief complaint, medical images and laboratory test results |
| BioMedCLIP | [A Multimodal Biomedical Foundation Model Trained from Fifteen Million Image–Text Pairs](https://ai.nejm.org/doi/full/10.1056/AIoa2400640) | [microsoft/BiomedCLIP_data_pipeline](https://github.com/microsoft/BiomedCLIP_data_pipeline) | [HF](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) | Pretrained on 15 million biomedical image–text pairs collected from 4.4 million scientific articles | 



## 3) Review paper


| Name      | Paper | GitHub | Hugging Face | Note |
|-----------|-------|--------|:------------:|------|
| GMAI | [Foundation models for generalist medical artificial intelligence]([https://ai.nejm.org/doi/full/10.1056/AIoa2300221](https://www.nature.com/articles/s41586-023-05881-4?fromPaywallRec=false#Abs1)) | - | — | - |
| GAI | [Generative artificial intelligence in medicine](https://www.nature.com/articles/s41591-025-03983-2#Fig1) | - | - | - |
