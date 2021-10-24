## Introduction

![model image](fig1.png)

Most electronic medical records, such as free-text radiological reports, are unstructured; however, the methodological approaches to analyzing these accumulating unstructured records are limited. This article proposes a deep-transfer-learning-based natural language processing model that analyzes serial magnetic resonance imaging reports of rectal cancer patients and predicts their overall survival. To evaluate the model, a retrospective cohort study of 4,338 rectal cancer patients was conducted. The experimental results revealed that the proposed model utilizing pre-trained clinical linguistic knowledge could predict the overall survival of patients without any structured information and was superior to the carcinoembryonic antigen in predicting survival. The deep-transfer-learning model using free-text radiological reports can predict the survival of patients with rectal cancer, thereby increasing the utility of unstructured medical big data.

## Demo

### Prerequisites
We provide toy data to run our implementation.
The full TCGA gene expression data can be downloaded in ICGC data portal.

https://dcc.icgc.org/


### Training and evaluating our deep learning based rectal cancer patient survival prediction model
```
python main.py
```

## Authors

* Sunkyu Kim
* Choong-kun Lee 
* Yonghwa Choi
* Eun Sil Baek
* Jeong Eun Choi
* Joon Seok Lim
* Jaewoo Kang
* Sang Joon Shin
