from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import shutil
import sys

import tensorflow as tf

import mysql.connector
import time

config = {
    'auth_plugin': 'mysql_native_password',
    'user': 'skerbey',
    'password': 'password',
    'host': 'localhost',
    'ssl_ca': 'C:\certificates\ca.pem',
    'ssl_cert': 'C:\certificates\client-cert.pem',
    'ssl_key': 'C:\certificates\client-key.pem',
}

parser = argparse.ArgumentParser()


parser.add_argument(
    '--model_dir', type=str, default='/tmp/dota_model',
    help='Base directory for the model.')

parser.add_argument(
    '--train_epochs', type=int, default=40, help='Number of training epochs.')

parser.add_argument(
    '--epochs_per_eval', type=int, default=4,
    help='The number of training epochs to run between evaluations.')

parser.add_argument(
    '--batch_size', type=int, default=40, help='Number of examples per batch.')

parser.add_argument(
    '--train', default=False, action='store_const', const=True,
)

parser.add_argument(
    '--predict', default=False, action='store_const', const=True,
)

_NUM_EXAMPLES = {
    'train': 80,
    'validation': 20,
}

cnx = None
cursor = None
num_heroes = 120


def build_model_columns():
    feature_columns = [tf.feature_column.numeric_column('x', shape=num_heroes*2)]

    return feature_columns


def build_estimator(model_dir):
    deep_columns = build_model_columns()
    hidden_units = [180]

    run_config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(device_count={'GPU': 0}))

    if FLAGS.train:
        return tf.estimator.DNNClassifier(
            model_dir=model_dir,
            feature_columns=deep_columns,
            hidden_units=hidden_units,
            config=run_config)
    else:
        return tf.estimator.DNNClassifier(
            model_dir=model_dir,
            feature_columns=deep_columns,
            hidden_units=hidden_units,
            config=run_config,
            warm_start_from=model_dir)


def get_sample_count():
    query = ("SELECT COUNT(*) FROM matches")
    cursor.execute(query)
    return cursor.fetchone()[0]


def setup_database():
    global cnx
    global cursor
    global num_heroes

    cnx = mysql.connector.connect(**config)
    cursor = cnx.cursor()
    query = ("USE dota2")
    cursor.execute(query)

    query = ("SELECT id FROM heroes")
    cursor.execute(query)
    h_max = -1
    for h in cursor.fetchall():
        if h[0] > h_max:
            h_max = h[0]
    print(h_max)
    num_heroes = h_max


def query_matches(train_test):
    if train_test == "test":
        query = ("SELECT * FROM matches ORDER BY match_id DESC LIMIT %s")
        data = (_NUM_EXAMPLES['validation'],)
        cursor.execute(query, data)
        print("Test set of size: " + str(_NUM_EXAMPLES['validation']))
    elif train_test == "train":
        query = ("SELECT * FROM matches ORDER BY match_id DESC LIMIT %s")
        data = (_NUM_EXAMPLES['validation'],)
        cursor.execute(query, data)
        max_id = cursor.fetchall()[-1:][0][0]
        print("Starting at match_id: " + str(max_id))
        query = ("SELECT * FROM matches WHERE match_id<%s ORDER BY match_id DESC LIMIT %s")
        data = (max_id,_NUM_EXAMPLES['train'])
        cursor.execute(query, data)
        print("Training set of size: " + str(_NUM_EXAMPLES['train']))
    else:
        print("wrong train_test value")
    matches = []
    m_columns = cursor.column_names
    for m in cursor.fetchall():
        m = dict(zip(m_columns, m))
        matches.append({'match_id': m['match_id'], 'radiant_win': m['radiant_win'], 'radiant': [], 'dire': []})
    match_hero_stmt = ("SELECT * FROM match_hero WHERE match_id=%s")
    for m in matches:
        data = (m['match_id'],)
        cursor.execute(match_hero_stmt,data)
        mh_columns = cursor.column_names
        for mh in cursor.fetchall():
            mh = dict(zip(mh_columns, mh))
            if mh['player_slot'] < 128:
                m['radiant'].append(mh)
            else:
                m['dire'].append(mh)
    return matches


def format_matches(matches):
    features = []
    labels = []
    count = 0
    for m in matches:
        r_indices = []
        d_indices = []
        for r in m['radiant']:
            r_indices.append(r['hero_id']-1)
        for d in m['dire']:
            d_indices.append(d['hero_id']-1 + num_heroes)
        depth = 2*num_heroes
        r_tensor = tf.one_hot(r_indices, depth, dtype=tf.int32)
        d_tensor = tf.one_hot(d_indices, depth, dtype=tf.int32)
        sess = tf.InteractiveSession()
        features.append(tf.convert_to_tensor(sum(r_tensor.eval())+sum(d_tensor.eval())))
        labels.append(tf.convert_to_tensor(m['radiant_win']))
        sess.close()
        count = count + 1
        if count % 100 == 0:
            print("Formatted " + str(count) + " matches.")
    features = {'x': features}
    return features, labels


def input_fn(num_epochs, shuffle, batch_size, train_test):

    def query_format_matches():
        print('Querying matches')
        matches = query_matches(train_test)
        print('Formatting matches')
        f, l = format_matches(matches)
        return f, l

    print("Creating dataset from tensor slices")
    dataset = tf.data.Dataset.from_tensor_slices(query_format_matches())

    if shuffle:
        dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)


    return dataset


# attempt to use a mapping function for data input
# def input_fn(num_epochs, shuffle, batch_size, train_test):
#
#     millis = int(round(time.time() * 1000))
#
#     def format_match(match):
#         r_indices = []
#         d_indices = []
#         for r in match['radiant']:
#             r_indices.append(r['hero_id']-1)
#         for d in match['dire']:
#             d_indices.append(d['hero_id']-1 + num_heroes)
#         depth = 2*num_heroes
#         r_tensor = tf.one_hot(r_indices, depth, dtype=tf.int32)
#         d_tensor = tf.one_hot(d_indices, depth, dtype=tf.int32)
#         sess = tf.InteractiveSession()
#         features = (tf.convert_to_tensor(sum(r_tensor.eval())+sum(d_tensor.eval())))
#         label = (tf.convert_to_tensor(match['radiant_win']))
#         sess.close()
#         features = {'x': features}
#         return features, label
#
#     print('Querying matches')
#     matches = query_matches(train_test)
#
#     dataset = tf.data.Dataset.from_tensor_slices(matches)
#
#     if shuffle:
#         dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])
#
#     dataset = dataset.map(format_match, num_parallel_calls=8)
#
#     dataset = dataset.repeat(num_epochs)
#     dataset = dataset.batch(batch_size)
#
#     millis2 = int(round(time.time() * 1000))
#
#     print('-' * 60)
#     print("input time")
#     print(millis2-millis)
#     print('-' * 60)
#
#     return dataset


def main(unused_argv):

    FLAGS.train_epochs = 3
    FLAGS.epochs_per_eval = 1

    setup_database()

    # num_samples = get_sample_count()
    num_samples = 250

    global _NUM_EXAMPLES
    _NUM_EXAMPLES = {
        'train': num_samples - int(0.2 * num_samples),
        'validation': int(0.2 * num_samples),
    }

    if not FLAGS.train and not FLAGS.predict:
        print('No training or prediction called')
        return

    if FLAGS.train:

        # Clean up the model directory if present
        shutil.rmtree(FLAGS.model_dir, ignore_errors=True)
        model = build_estimator(FLAGS.model_dir)

        millis = int(round(time.time() * 1000))

        # Train and evaluate the model every `FLAGS.epochs_per_eval` epochs.
        for n in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
            model.train(input_fn=lambda: input_fn(
                FLAGS.epochs_per_eval, True, FLAGS.batch_size, "train"))

            results = model.evaluate(input_fn=lambda: input_fn(
                1, False, FLAGS.batch_size, "test"))

            # Display evaluation metrics
            print('Results at epoch', (n + 1) * FLAGS.epochs_per_eval)
            print('-' * 60)

            for key in sorted(results):
                print('%s: %s' % (key, results[key]))

        millis2 = int(round(time.time() * 1000))

        print('-' * 60)
        print("input time")
        print(millis2-millis)
        print('-' * 60)

    else:
        model = build_estimator(FLAGS.model_dir)

    if FLAGS.predict:
        # TODO create input argument to predict against and run it here
        # predictions = model.predict(input_fn=lambda: input_fn(
        #    1, False, FLAGS.batch_size))
        return


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)