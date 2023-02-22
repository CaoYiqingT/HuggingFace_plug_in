# Huggingface_Multitast_Transformers
This project implements the multitask version of the Huggingface transformers.
The models in the repository can be used in almost the same way as the models that come with Huggingface, and you can initialize them in the following ways:
> Noted that our models can be initialized with Huggingface models 
```
model = RobertaForMultipleTask.from_pretrained(model_name_or_path)
model = RobertaForMultipleTask(config)
```
You can save them in following way:
```
RobertaForMultipleTask.save_pretrained(save_path)
```
