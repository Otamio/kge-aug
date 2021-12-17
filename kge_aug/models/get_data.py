import pykeen
from pykeen.triples import TriplesFactory
from kge_aug.models import constants


dataset_path_mapping = {
    'fb15k237': constants.dataset_fb15k237,
    'yago15k': constants.dataset_yago15k
}


def get(dataset, training_fname):

    dataset_path = dataset_path_mapping[dataset]

    if pykeen.get_version() == "1.0.0":

        training = TriplesFactory(path=f"{dataset_path}/{training_fname}")
        testing = TriplesFactory(
            path=f"{dataset_path}/test.tsv",
            entity_to_id=training.entity_to_id,
            relation_to_id=training.relation_to_id
        )
        validation = TriplesFactory(
            path=f"{dataset_path}/valid.tsv",
            entity_to_id=training.entity_to_id,
            relation_to_id=training.relation_to_id
        )

    else:

        training = TriplesFactory.from_path(f"{dataset_path}/{training_fname}")
        testing = TriplesFactory.from_path(
            f"{dataset_path}/test.tsv",
            entity_to_id=training.entity_to_id,
            relation_to_id=training.relation_to_id
        )
        validation = TriplesFactory.from_path(
            f"{dataset_path}/valid.tsv",
            entity_to_id=training.entity_to_id,
            relation_to_id=training.relation_to_id
        )

    return training, testing, validation
