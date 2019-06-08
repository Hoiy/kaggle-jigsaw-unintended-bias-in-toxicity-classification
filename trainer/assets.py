import os
import subprocess
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("output_dir", "assets", "The output directory.")

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.gfile.MakeDirs(FLAGS.output_dir)

    subprocess.run([
        'kaggle', 'competitions', 'download',
        'jigsaw-unintended-bias-in-toxicity-classification',
        '-p', FLAGS.output_dir
    ])

    subprocess.run(['unzip', '-d', {FLAGS.output_dir}, f'{FLAGS.output_dir}/sample_submission.csv.zip'])
    subprocess.run(['unzip', '-d', {FLAGS.output_dir}, f'{FLAGS.output_dir}/test.csv.zip'])
    subprocess.run(['unzip', '-d', {FLAGS.output_dir}, f'{FLAGS.output_dir}/train.csv.zip'])
    subprocess.run(['rm', '-r', f'{FLAGS.output_dir}/*.zip'])


if __name__ == "__main__":
    tf.app.run()
