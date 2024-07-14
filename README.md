# Rice Classifier

Interactive AI classifier of rice grain.

## Demo
https://rice-classifier.streamlit.app<br>
![Streamlit App Demo](docs/images/demo.png "Streamlit App Demo")

## ML Workflow
![Machine Learning Workflow Diagram](docs/images/ml_workflow_diagram.png "Machine Learning Workflow Diagram")

# Metrics
								
| Method | Accuracy (train) | Precision (train) | Recall (train) | F1 (train) | AUC (train) | Accuracy (test) | Precision (test) | Recall (test) | F1 (test) | AUC (test) |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| logreg_sag | 0.926	| 0.889	|0.945	|0.916	|0.978	|0.911	|0.875	|0.923	|0.899	|0.975|
| logreg_lbfgs | 0.926|	0.888	|**0.946**	|0.916	|0.978	|0.912	|0.878	|0.923	|0.900	|0.975|
| **svm_linear** | 0.927	|0.894	|0.941	|0.917	|0.979	|**0.923**	|0.899	|**0.923**	|**0.911**	|0.976|
| svm_poly | 0.920	|**0.910**	|0.903	|0.906	|0.971	|0.908	|0.898	|0.887	|0.892	|**0.968**|
| nn       | **0.931** |	0.908	|0.934	|**0.921**	|**0.980**	|0.921	|**0.910**	|0.905	|0.908	|0.976|

