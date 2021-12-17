import pykeen
from pykeen.triples import TriplesFactory
from kge_aug.models import constants


dataset_path_mapping = {
    'fb15k237': constants.dataset_fb15k237,
    'yago15k': constants.dataset_yago15k
}

target_path_mapping = {
    'vanilla': constants.link_prediction_path,
    'lp': constants.link_prediction_path_processed,
    'np': constants.numeric_prediction_path
}


def get(dataset, target, training_fname):

    dataset_path = dataset_path_mapping[dataset]
    target_path = target_path_mapping[target]

    if pykeen.get_version() == "1.0.0":

        training = TriplesFactory(path=f"{dataset_path}/{target_path}/{training_fname}")
        if target != "np":
            validation = TriplesFactory(
                path=f"{dataset_path}/data/valid.tsv",
                entity_to_id=training.entity_to_id,
                relation_to_id=training.relation_to_id
            )
            testing = TriplesFactory(
                path=f"{dataset_path}/data/test.tsv",
                entity_to_id=training.entity_to_id,
                relation_to_id=training.relation_to_id
            )
        else:
            validation = TriplesFactory(
                path=f"{dataset_path}/numeric/dummy.tsv",
                entity_to_id=training.entity_to_id,
                relation_to_id=training.relation_to_id
            )
            testing = TriplesFactory(
                path=f"{dataset_path}/numeric/dummy.tsv",
                entity_to_id=training.entity_to_id,
                relation_to_id=training.relation_to_id
            )

    else:

        training = TriplesFactory.from_path(f"{dataset_path}/{target_path}/{training_fname}")
        if target != "np":
            testing = TriplesFactory.from_path(
                f"{dataset_path}/data/test.tsv",
                entity_to_id=training.entity_to_id,
                relation_to_id=training.relation_to_id
            )
            validation = TriplesFactory.from_path(
                f"{dataset_path}/data/valid.tsv",
                entity_to_id=training.entity_to_id,
                relation_to_id=training.relation_to_id
            )
        else:
            validation = TriplesFactory.from_path(
                f"{dataset_path}/numeric/dummy.tsv",
                entity_to_id=training.entity_to_id,
                relation_to_id=training.relation_to_id
            )
            testing = TriplesFactory.from_path(
                f"{dataset_path}/numeric/dummy.tsv",
                entity_to_id=training.entity_to_id,
                relation_to_id=training.relation_to_id
            )

    return training, testing, validation
