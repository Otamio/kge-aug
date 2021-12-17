from pykeen.pipeline import pipeline
from kge_aug.models import constants


def get_pipeline(training):

    return pipeline(
        training_triples_factory=training,
        dataset_kwargs=dict(
            create_inverse_triples=False
        ),
        model='TransE',
        model_kwargs=dict(
            embedding_dim=128,
            scoring_fct_norm=1,
            automatic_memory_optimization=True
        ),
        training_loop='sLCWA',
        training_kwargs=dict(
            num_epochs=constants.epochs,
            batch_size=512,
            label_smoothing=0.0
        ),
        loss="marginranking",
        loss_kwargs=dict(
            reduction="mean",
            margin=2.6861406188094135
        ),
        optimizer='Adam',
        optimizer_kwargs=dict(
            lr=0.0011936858534857958
        ),
        negative_sampler='basic',
        negative_sampler_kwargs=dict(
            num_negs_per_pos=80
        ),
        regularizer='no',
    )
