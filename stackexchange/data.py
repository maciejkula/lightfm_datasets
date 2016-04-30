import argparse
import array
import codecs
import datetime
import os
import subprocess
import sys
import time

from lxml import etree

import numpy as np

import progressbar

import requests

import scipy.sparse as sp


def _process_post_tags(tags_string):

    return [x for x in tags_string.replace('<', ' ').replace('>', ' ').split(' ') if x]


def _read_interactions(post_data):

    for line in post_data:
        try:
            datum = dict(etree.fromstring(line).items())
        except etree.XMLSyntaxError:
            continue

        is_answer = datum.get('ParentId') is not None

        if not is_answer:
            continue

        try:
            user_id = int(datum['OwnerUserId'])
            question_id = int(datum['ParentId'])
            time_created = time.mktime(datetime.datetime
                                       .strptime(datum['CreationDate'],
                                                 '%Y-%m-%dT%H:%M:%S.%f')
                                       .timetuple())
        except KeyError:
            continue

        if user_id == -1:
            continue

        yield user_id, question_id, time_created


def _read_question_features(post_data):

    for line in post_data:

        try:
            datum = dict(etree.fromstring(line).items())
        except etree.XMLSyntaxError:
            continue

        is_question = datum.get('ParentId') is None

        if not is_question:
            continue

        question_id = int(datum['Id'])
        question_tags = _process_post_tags(datum.get('Tags', ''))

        yield question_id, question_tags


class IncrementalSparseMatrix(object):

    def __init__(self):

        self.row = array.array('i')
        self.col = array.array('i')
        self.data = array.array('f')

    def append(self, row, col, data):

        self.row.append(row)
        self.col.append(col)
        self.data.append(data)

    def tocoo(self, shape=None):

        row = np.array(self.row, dtype=np.int32)
        col = np.array(self.col, dtype=np.int32)
        data = np.array(self.data, dtype=np.float32)

        if shape is None:
            shape = (row.max() + 1, col.max() + 1)

        return sp.coo_matrix((data, (row, col)),
                             shape=shape)


class Dataset(object):

    def __init__(self):

        # Mappings
        self.user_mapping = {}
        self.question_mapping = {}
        self.tag_mapping = {}

        # Question features
        self.question_features = IncrementalSparseMatrix()

        # User-question matrix
        self.interactions = IncrementalSparseMatrix()

    def fit_features_matrix(self, question_features):

        self.question_features = IncrementalSparseMatrix()

        for (question_id, question_tags) in question_features:

            translated_question_id = (self.question_mapping
                                      .setdefault(question_id,
                                                  len(self.question_mapping)))

            for tag in question_tags:
                translated_tag_id = (self.tag_mapping
                                     .setdefault(tag,
                                                 len(self.tag_mapping)))

                self.question_features.append(translated_question_id,
                                              translated_tag_id,
                                              1.0)

    def fit_interaction_matrix(self, interactions):

        self.interactions = IncrementalSparseMatrix()

        for (user_id, question_id, timestamp) in interactions:

            translated_user_id = (self.user_mapping
                                  .setdefault(user_id,
                                              len(self.user_mapping)))
            translated_question_id = (self.question_mapping
                                      .setdefault(question_id,
                                                  len(self.question_mapping)))

            self.interactions.append(translated_user_id,
                                     translated_question_id,
                                     timestamp)

    def get_features_matrix(self):

        return self.question_features.tocoo(shape=(len(self.question_mapping),
                                                   len(self.tag_mapping)))

    def get_interaction_matrix(self):

        return self.interactions.tocoo(shape=(len(self.user_mapping),
                                              len(self.question_mapping)))

    def get_feature_labels(self):

        tags = sorted(self.tag_mapping.items(), key=lambda x: x[1])

        return np.array([x[0] for x in tags], dtype=np.dtype('|U50'))


def serialize_data(file_path, interactions, features, labels):

    arrays = {}

    for name, mat in (('interactions', interactions),
                      ('features', features)):
        arrays['{}_{}'.format(name, 'shape')] = (np.array(mat.shape,
                                                          dtype=np.int32)
                                                 .flatten()),
        arrays['{}_{}'.format(name, 'row')] = mat.row
        arrays['{}_{}'.format(name, 'col')] = mat.col
        arrays['{}_{}'.format(name, 'data')] = mat.data

    arrays['labels'] = labels

    np.savez_compressed(file_path, **arrays)


def read_data(data_path):
    """
    Construct a user-thread matrix, where a user interacts
    with a thread if they post an answer in it.
    """

    dataset = Dataset()

    with codecs.open(data_path, 'r', encoding='utf-8') as data_file:
        dataset.fit_features_matrix(_read_question_features(data_file))
    with codecs.open(data_path, 'r', encoding='utf-8') as data_file:
        dataset.fit_interaction_matrix(_read_interactions(data_file))

    question_features = dataset.get_features_matrix()
    interactions = dataset.get_interaction_matrix()
    feature_labels = dataset.get_feature_labels()

    assert question_features.shape[0] == interactions.shape[1]
    assert question_features.shape[1] == len(feature_labels)

    return interactions, question_features, feature_labels


def get_data(dataset_name, url, data_root):

    data_dir = os.path.join(data_root, dataset_name)

    if not os.path.exists(data_dir):
        print('Creating data dir...')
        os.makedirs(data_dir)

    archive_path = os.path.join(data_dir, 'archive.7z')

    if not os.path.isfile(archive_path):
        print('Downloading data...')
        chunk_size = 2 * 2**20

        response = requests.get(url, stream=True)
        total_length = int(response.headers.get('content-length'))
        chunks = total_length / chunk_size

        with open(archive_path, 'wb') as fd:
            bar = progressbar.ProgressBar(max_value=chunks)
            for chunk in bar(response.iter_content(chunk_size=chunk_size)):
                fd.write(chunk)

    posts_path = os.path.join(data_dir, 'Posts.xml')

    if not os.path.isfile(posts_path):
        print('Extracting data...')

        with open(os.devnull, 'w') as fnull:
            try:
                subprocess.check_call(['7za', 'x', archive_path],
                                      cwd=data_dir, stdout=fnull)
            except (OSError, subprocess.CalledProcessError):
                raise Exception('You must install p7zip to extract the data.')

    return posts_path


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='Name of the dataset to process.')

    args = parser.parse_args()

    if args.dataset == 'crossvalidated':
        url = 'http://archive.org/download/stackexchange/stats.stackexchange.com.7z'
    elif args.dataset == 'stackoverflow':
        url = 'https://archive.org/download/stackexchange/stackoverflow.com-Posts.7z'
    else:
        raise ValueError('Unknown dataset')

    posts_path = get_data(args.dataset, url, os.path.dirname(sys.argv[0]))

    print('Reading data...')
    interactions, features, labels = read_data(posts_path)

    output_fname = 'stackexchange_{}.npz'.format(args.dataset)

    print('Writing output...')
    output_path = os.path.join(os.path.dirname(posts_path), output_fname)
    serialize_data(output_path, interactions, features, labels)

    print('Done.')
