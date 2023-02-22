# Huggingface_Multitast_Transformers
This is a huggingface transformer that implements Multitask.
The models in the repository can be used in almost the same way as the models that come with Huggingface, and you can initialize them in the following way:
```
// Init a model with RoBERTa parameters
model = RobertaForMultipleTask.from_pretrained("roberta-base")
//Init a model with random parameters
config = AutoConfig.from_pretrained("roberta-base")
model = RobertaForMultipleTask(config)
```
