## **[SIB-200: A Simple, Inclusive, and Big Evaluation Dataset for Topic Classification in 200+ Languages and Dialects](https://arxiv.org/abs/2309.07445)**

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

## BibTeX entry and citation info


```
@misc{adelani2023sib200,
      title={SIB-200: A Simple, Inclusive, and Big Evaluation Dataset for Topic Classification in 200+ Languages and Dialects}, 
      author={David Ifeoluwa Adelani and Hannah Liu and Xiaoyu Shen and Nikita Vassilyev and Jesujoba O. Alabi and Yanke Mao and Haonan Gao and Annie En-Shiun Lee},
      year={2023},
      eprint={2309.07445},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
