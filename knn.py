"""Lens training + nearest neighbors classification pipeline."""

import os

import sys
sys.path.insert(1, 'google_research/')

import flax
from flax import nn
from flax.training import checkpoints

import jax
from jax import random
import jax.numpy as jnp
from jax.config import config
config.enable_omnistaging()

import numpy as np

import pandas as pd

import json

import copy

import time

from pkg_resources import resource_filename

from fs_gcsfs import GCSFS

from google_research.protein_lm import domains, models

from contextual_lenses.contextual_lenses import reduce_fn_name_to_fn

from contextual_lenses.train_utils import create_optimizer, train, \
create_representation_model, create_transformer_representation_model, \
architecture_to_layers

from contextual_lenses.encoders import encoder_fn_name_to_fn

from contextual_lenses.loss_fns import cross_entropy_loss

from contextual_lenses.pfam_utils import get_family_ids, PFAM_NUM_CATEGORIES, \
pfam_evaluate, create_pfam_batches, pfam_nearest_neighbors_classification

from contextual_lenses.load_transformer import load_transformer_params

from absl import app, flags

# Define flags.
FLAGS = flags.FLAGS

flags.DEFINE_string('encoder_fn_name', 'cnn_one_hot',
                    'Name of encoder_fn to use. None if using Transformer.')
flags.DEFINE_string('encoder_fn_kwargs_path', 'cnn_kwargs',
                    'Path to encoder_fn_kwargs.')
flags.DEFINE_string('reduce_fn_name', 'linear_max_pool',
                    'Name of reduce_fn to use.')
flags.DEFINE_string('reduce_fn_kwargs_path', 'linear_pool_1024',
                    'Path to reduce_fn_kwargs.')

flags.DEFINE_integer('knn_batch_size', 64,
                     'Batch size for KNN vector computation.')

flags.DEFINE_integer('first_test_family', 15001, 'First family to test on.')
flags.DEFINE_integer('last_test_family', 16000, 'Last family to test on.')

flags.DEFINE_integer('lens_shuffle_seed', 0,
                     'Random seed used for lens training data batching.')
flags.DEFINE_integer('lens_sample_random_state', 0,
                     'Random state used for lens training data sampling.')
flags.DEFINE_integer('knn_shuffle_seed', 1,
                     'Random seed used for KNN data batching.')
flags.DEFINE_integer('knn_sample_random_state', 1,
                     'Random state used for KNN data sampling.')
flags.DEFINE_integer('model_random_key', 0,
                     'Random key used for model instantiation.')

flags.DEFINE_boolean('use_transformer', False,
                     'Whether or not to use transformer encoder')
flags.DEFINE_boolean('use_bert', False,
                     'Whether or not to use bidirectional transformer.')
flags.DEFINE_string('restore_transformer_dir', None,
                    'Directory to load pretrained transformer from.')

flags.DEFINE_string('gcs_bucket', 'neuralblast',
                    'GCS bucket to save to and load from.')
flags.DEFINE_string('data_partitions_dirpath', 'random_split/',
                    'Location of Pfam data in GCS bucket.')


def get_model_kwargs(encoder_fn_name, encoder_fn_kwargs_path, reduce_fn_name,
                     reduce_fn_kwargs_path):
    """Determines model components using string names."""

    encoder_fn = encoder_fn_name_to_fn(encoder_fn_name)
    encoder_fn_kwargs = json.load(
        open(
            resource_filename(
                'contextual_lenses.resources',
                os.path.join('encoder_fn_kwargs_resources',
                             encoder_fn_kwargs_path + '.json'))))

    reduce_fn = reduce_fn_name_to_fn(reduce_fn_name)
    reduce_fn_kwargs = json.load(
        open(
            resource_filename(
                'contextual_lenses.resources',
                os.path.join('reduce_fn_kwargs_resources',
                             reduce_fn_kwargs_path + '.json'))))

    layers, trainable_encoder = architecture_to_layers(encoder_fn_name,
                                                       reduce_fn_name)

    return encoder_fn, encoder_fn_kwargs, reduce_fn, reduce_fn_kwargs, layers


def create_model(encoder_fn,
                 encoder_fn_kwargs,
                 reduce_fn,
                 reduce_fn_kwargs,
                 layers,
                 output='prediction',
                 use_transformer=False,
                 use_bert=False,
                 restore_transformer_dir=None,
                 encoder_fn_params=None,
                 reduce_fn_params=None,
                 predict_fn_params=None,
                 random_key=0):
    """Creates representation model (encoder --> lens --> predictor) architecture."""

    family_ids = get_family_ids()
    num_families = len(family_ids)

    if use_transformer:

        if use_bert:
            model_cls = models.FlaxBERT
        else:
            model_cls = models.FlaxLM

        if encoder_fn_params is not None:
            pretrained_transformer_params = encoder_fn_params
        else:
            if restore_transformer_dir is not None:
                pretrained_transformer_params = load_transformer_params(
                    restore_transformer_dir, model_cls)
            else:
                pretrained_transformer_params = None

        model = create_transformer_representation_model(
            transformer_kwargs=encoder_fn_kwargs,
            reduce_fn=reduce_fn,
            reduce_fn_kwargs=reduce_fn_kwargs,
            num_categories=PFAM_NUM_CATEGORIES,
            output_features=num_families,
            bidirectional=use_bert,
            output=output,
            key=random.PRNGKey(random_key),
            encoder_fn_params=pretrained_transformer_params,
            reduce_fn_params=reduce_fn_params,
            predict_fn_params=predict_fn_params)

    else:
        model = create_representation_model(
            encoder_fn=encoder_fn,
            encoder_fn_kwargs=encoder_fn_kwargs,
            reduce_fn=reduce_fn,
            reduce_fn_kwargs=reduce_fn_kwargs,
            num_categories=PFAM_NUM_CATEGORIES,
            output_features=num_families,
            output=output,
            key=random.PRNGKey(random_key),
            encoder_fn_params=encoder_fn_params,
            reduce_fn_params=reduce_fn_params,
            predict_fn_params=predict_fn_params)

    return model


def set_model_parameters(model, params):
    """Updates a model's parameters using a parameters dictionary."""

    params = copy.deepcopy(params)

    assert (
        model.params.keys() == params.keys()), 'Model parameters do not match!'

    for layer in model.params.keys():
        model.params[layer] = params[layer]

    return model


def measure_nearest_neighbor_performance(accuracy_label, encoder,
                                         family_accessions, batch_size,
                                         train_samples, shuffle_seed,
                                         sample_random_state):
    """Measures nearest neighbor classification performance and updates datum."""

    results = pfam_nearest_neighbors_classification(
        encoder=encoder,
        family_accessions=family_accessions,
        batch_size=batch_size,
        train_samples=train_samples,
        shuffle_seed=shuffle_seed,
        sample_random_state=sample_random_state,
        data_partitions_dirpath=FLAGS.data_partitions_dirpath,
        gcs_bucket=FLAGS.gcs_bucket)[0]

    accuracy = results['1-nn accuracy']

    accuracy_dict = {accuracy_label: accuracy}

    return accuracy_dict


# Train lens and measure performance of lens and nearest neighbors classifier.
def main(_):

    if FLAGS.use_transformer:
        assert (
            FLAGS.encoder_fn_name == 'transformer'
        ), 'encoder_fn_name must be transformer if use_transformer is True!'

    gcsfs = GCSFS(FLAGS.gcs_bucket)

    knn_train_samples_ = [1, 5, 10, 50]

    family_ids = get_family_ids()
    num_families = len(family_ids)
    loss_fn_kwargs = {'num_classes': num_families}

    knn_test_family_accessions = []
    for _ in range(FLAGS.first_test_family, FLAGS.last_test_family + 1):
        family_name = 'PF%05d' % _
        knn_test_family_accessions.append(family_name)

    encoder_fn, encoder_fn_kwargs, reduce_fn, reduce_fn_kwargs, layers = get_model_kwargs(
        encoder_fn_name=FLAGS.encoder_fn_name,
        encoder_fn_kwargs_path=FLAGS.encoder_fn_kwargs_path,
        reduce_fn_name=FLAGS.reduce_fn_name,
        reduce_fn_kwargs_path=FLAGS.reduce_fn_kwargs_path)

    embedding_model = create_model(
        encoder_fn=encoder_fn,
        encoder_fn_kwargs=encoder_fn_kwargs,
        reduce_fn=reduce_fn,
        reduce_fn_kwargs=reduce_fn_kwargs,
        layers=layers,
        output='embedding',
        use_transformer=FLAGS.use_transformer,
        use_bert=FLAGS.use_bert,
        restore_transformer_dir=FLAGS.restore_transformer_dir,
        random_key=FLAGS.model_random_key)

    knn_results = []

    for knn_train_samples in knn_train_samples_:

        knn_results.append(
            measure_nearest_neighbor_performance(
                accuracy_label='test_knn_accuracy_untrained_lens_' +
                str(knn_train_samples) + '_knn_train_samples',
                encoder=embedding_model,
                family_accessions=knn_test_family_accessions,
                batch_size=FLAGS.knn_batch_size,
                train_samples=knn_train_samples,
                shuffle_seed=FLAGS.knn_shuffle_seed,
                sample_random_state=FLAGS.knn_sample_random_state))

    print(knn_results)

    
if __name__ == '__main__':
    app.run(main)