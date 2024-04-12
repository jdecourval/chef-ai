# chef-ai

Download here: https://huggingface.co/jdecourval/chef-ai

This repository is meant as an exercise for finetuning a transformer based large language model. 
The goal is to finetune a small model as a cooking assistant, and be able to do so on a single 24GB.
For an extra challenge, I used an AMD GPU (7900xtx).

All finetuning steps are demonstrated. The application:
- Scrapes data sources on the internet.
- Index that information into a sqlite database.
- Uses llm based subpipelines to generate a chat-formatted dataset from the database.
- Performs efficient supervised finetuning from the dataset.
- Merges the resulting LoRA into the base model.

To avoid being blamed for scrapping, the source's base url (of the format https://mysource.com) is not included in this repo, and must be specified through the `SPIDER_BASE_URL` environment variable, or [spider.py](spider/spider.py) must be adapted for your own different source. 
You must figure this out yourself.

To execute, install the requirements and run `main.py` followed by the path to an inference model supported by llama.cpp or exllama. Don't forget to define `SPIDER_BASE_URL` as explained above. To use llama.cpp's server, you must also define the `LLAMACPP_BIN_PATH` environment variable to point to llama.cpp's bin folder.
Example:
```shell
export SPIDER_BASE_URL="https://redacted.com"
export LLAMACPP_BIN_PATH="/home/me/llamacpp/bin"
source venv/bin/activate
python main.py inference-model.gguf
```

The whole thing takes a few days to complete on my computer, but every step is resumable. In other words, the process can be interrupted at any time, and will (roughly) pick up where it was when it is restarted.
With a single 24GB GPU, you can really only expect to be able to finetune 7B parameters models.
You can use something larger for the inference model however, but keep in mind the application will probably use the model close to 10000 times over its entire course, so it needs to be fast. Smaller model are faster not only because of their lower compute requirements, but also because this repository uses continuous batching.