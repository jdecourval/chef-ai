# chef-ai
This repository is meant as an exercise for finetuning a transformer based large language model. 
The goal is to finetune a small model as a cooking assistant, and be able to do so on a single 24GB.
For an extra challenge, I used an AMD GPU (7900xtx).

All finetuning steps are demonstrated. The application:
- Scrapes data sources on the internet.
- Index that information into a sqlite database.
- Uses llm based subpipelines to generate chat-formatted datasets from the database.
- Performs efficient supervised finetuning from the datasets.
- Merges the resulting LoRA into the base model.

To avoid being blamed from scrapping, the source's base url (of the format https://mysource.com) is not included in this repo, and must be specified through the `SPIDER_BASE_URL` environment variable, or [spider.py](spider/spider.py) must be adapted for your own different source. 
You must figure this out yourself.

To execute, install the requirements and execute `main` followed by the path to an inference model supported by llama.cpp or exllama. Don't forget to define `SPIDER_BASE_URL` as explained above. To use llama.cpp's server, you must also define the `LLAMACPP_BIN_PATH` environment variable to point to llama.cpp's bin folder.

The whole thing takes a few days to complete on my computer, but every step is resumable. In other words, the process can be interupted at any time, and will (roughly) pick up where it was when it is restarted.
