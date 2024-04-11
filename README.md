# chef-ai
This repository is meant as an exercise for finetuning a transformer based large language model. 
The goal is to finetune a small model as a cooking assistant, and be able to do so on a single 24GB.
For an extra challenge, I used an AMD GPU (7900xtx).

All steps are demonstrated. The application:
- Scrapes data sources on the internet.
- Index that information into a sqlite database.
- Uses llm based subpipelines to generate chat-formatted datasets from the database.
- Performs efficient supervised finetuning from the datasets.
- Merges the resulting LoRA into the base model.

To avoid being blamed from scrapping, the source's base url (of the format https://mysource.com) is not included in this repo, and must be specified through the `SPIDER_BASE_URL` environment variable, or [spider.py](spider/spider.py) must be adapted for your own different source. 
You must figure this out yourself.

To execute, install the requirements and execute main followed by the path to inference model supported by llama.cpp or exllama.