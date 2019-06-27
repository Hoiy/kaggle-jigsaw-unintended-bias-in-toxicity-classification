import os
import sys
from src.processor import JigsawUBITCProcessor
if not 'bert' in sys.path:
    sys.path += ['bert']
from run_classifier import *
import tensorflow as tf
# import pandas as pd
# from tqdm import tqdm
# from sklearn.model_selection import train_test_split
# from trainer.transform import parse_truths, bert_input_to_tfexample
# from berserker.transform import preprocess

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("assets_dir", "assets", "The assets directory generated by assets.py.")
flags.DEFINE_string("output_dir", "dataset", "The output directory.")
flags.DEFINE_integer("max_seq_length", 512, "Maximum sequence length.")
flags.DEFINE_integer("random_seed", 42, "Random seed.")

class flags:
    assets_dir = './assets'
    output_dir = 'dataset'
    vocab_file = 'assets/wwm_uncased_L-24_H-1024_A-16/vocab.txt'
    max_seq_length = 256
    do_lower_case = True

FLAGS = flags()

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.gfile.MakeDirs(FLAGS.output_dir)

    processor = JigsawUBITCProcessor()
    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    train_examples = processor.get_train_examples(FLAGS.assets_dir)    
    file_based_convert_examples_to_features(train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)

    test_file = os.path.join(FLAGS.output_dir, "test.tf_record")
    test_examples = processor.get_test_examples(FLAGS.assets_dir)
    file_based_convert_examples_to_features(test_examples, label_list, FLAGS.max_seq_length, tokenizer, test_file)


if __name__ == "__main__":
  tf.app.run()


import pandas as pd
df = pd.read_csv('assets/train.csv')

%mathlibplot inline
(df.target >= 0.5).describe()
1660540 / 1804874

df['l'] = df.comment_text.apply(lambda x: len(x.split(' ')))
df['l'].plot.hist(bins=100)
