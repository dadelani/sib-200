## **SIB-200: A Simple, Inclusive, and Big Evaluation Dataset for Topic Classification in 200+ Languages and Dialects**

This repository contains the [annotated English dataset](https://github.com/dadelani/sib-200/tree/main/data/eng), the [script](https://github.com/dadelani/sib-200/blob/main/get_flores_and_annotate.sh) to extend annotation to other languages and [code](https://github.com/dadelani/sib-200/tree/main/code) to run baseline text classification models. 


### Required dependencies
* python
  * [transformers](https://pypi.org/project/transformers/) : state-of-the-art Natural Language Processing for TensorFlow 2.0 and PyTorch.
  * sklearn
  * evaluate
  * datasets
  * pandas

```bash
pip install -r code/requirements.txt
```

### Create SIB dataset
```
sh get_flores_and_annotate.sh
```

### Run our baseline model using XLM-R
```
cd code/
sh xlmr_all.sh
```
