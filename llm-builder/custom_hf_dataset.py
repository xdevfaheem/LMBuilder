
import ftfy.bad_codecs
from datasets import Dataset, DatasetDict

print("Preparing Train Set...")
train = open('/kaggle/working/TinyStoriesV2-GPT4-train.txt', 'r', encoding='sloppy-windows-1252').read()
train = train.split('<|endoftext|>')
train = [l.strip() for l in train]

print("Preparing Validation Set...")
valid = open('/kaggle/working/TinyStoriesV2-GPT4-valid.txt', 'r', encoding='sloppy-windows-1252').read()
valid = valid.split('<|endoftext|>')
valid = [l.strip() for l in valid]

dataset = DatasetDict({
    'train': Dataset.from_dict({'text': train }),
    'validation': Dataset.from_dict({'text': valid}),
})
dataset.save_to_disk('/kaggle/working/datasets/tinystories/hf_dataset')
