# Huggingface_Multitast_Transformers
This is a huggingface transformer that implements Multitask.
The models in the repository can be used in almost the same way as the models that come with Huggingface, and you can initialize them in the following way:
> Noted that our models can be initialized with Huggingface models 
```
model = RobertaForMultipleTask.from_pretrained(model_name_or_path)
model = RobertaForMultipleTask(config)
```
You can save them in following way:
```
RobertaForMultipleTask.save_pretrained(save_path)
```
