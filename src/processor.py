import sys
import os
if not 'bert' in sys.path:
    sys.path += ['bert']
from run_classifier import *
import tokenization
import pandas as pd
from tqdm import tqdm

class JigsawUBITCProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
        # train = df.sample(frac=0.8, random_state=42)
        # df = train
        neg = df[df.target < 0.5]
        pos = df[df.target >= 0.5].sample(len(neg), replace=True, random_state=999)
        df = pd.concat([pos, neg]).sample(frac=1, random_state=888)
        return self._create_examples(df, "train")

    def get_dev_examples(self, data_dir):
        df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
        train = df.sample(frac=0.8, random_state=42)
        df = df[~df.index.isin(train.index)]

        pos = df[df.target >= 0.5]
        neg = df[df.target < 0.5].sample(n=len(pos))
        df = pd.concat([pos, neg]).sample(frac=1)
        return self._create_examples(df, "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            pd.read_csv(os.path.join(data_dir, 'test.csv')), "test")

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, df, set_type):
        examples = []
        if set_type == 'train':
            df = df.sample(frac=1).reset_index(drop=True)

        for i, row in tqdm(df.iterrows(), total=len(df)):
            guid = "%s-%s" % (set_type, i)
            if set_type == "test":
                text_a = tokenization.convert_to_unicode(row['comment_text'])
                label = "0"
            else:
                text_a = tokenization.convert_to_unicode(row['comment_text'])
                label = "1" if row['target'] >= 0.5 else "0"
            examples.append(
              InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples
