import os
import subprocess
import tensorflow as tf
from src.utils import maybe_download_unzip

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("output_dir", "assets", "The output directory.")

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.gfile.MakeDirs(FLAGS.output_dir)

    # subprocess.run([
    #     'kaggle', 'competitions', 'download',
    #     'jigsaw-unintended-bias-in-toxicity-classification',
    #     '-p', FLAGS.output_dir
    # ])

    subprocess.run(['unzip', '-d', FLAGS.output_dir, f'{FLAGS.output_dir}/sample_submission.csv.zip'])
    subprocess.run(['unzip', '-d', FLAGS.output_dir, f'{FLAGS.output_dir}/test.csv.zip'])
    subprocess.run(['unzip', '-d', FLAGS.output_dir, f'{FLAGS.output_dir}/train.csv.zip'])
    subprocess.run(['rm', '-r', f'{FLAGS.output_dir}/*.zip'])

    os.chmod(f"{FLAGS.output_dir}/train.csv", 0o644)
    os.chmod(f"{FLAGS.output_dir}/test.csv", 0o664)
    os.chmod(f"{FLAGS.output_dir}/sample_submission.csv", 0o664)

    # maybe_download_unzip(
    #     'https://storage.googleapis.com/bert_models/2019_05_30/wwm_cased_L-24_H-1024_A-16.zip',
    #     FLAGS.output_dir
    # )
    #
    # maybe_download_unzip(
    #     'https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip',
    #     FLAGS.output_dir
    # )


if __name__ == "__main__":
    tf.app.run()
