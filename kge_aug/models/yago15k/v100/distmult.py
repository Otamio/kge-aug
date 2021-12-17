from pykeen.pipeline import pipeline
from kge_aug.models import constants


def get_pipeline(training, testing, validation):

    return pipeline(
        training_triples_factory=training,
        testing_triples_factory=testing,
        validation_triples_factory=validation,
        dataset_kwargs=dict(
            create_inverse_triples=True
        ),
        model='DistMult',
        model_kwargs=dict(
            embedding_dim=256,
            automatic_memory_optimization=True
        ),
        training_loop='LCWA',
        training_kwargs=dict(
            num_epochs=60,
            batch_size=256,
            label_smoothing=0.08094657004944494
        ),
        loss="crossentropy",
        optimizer='Adam',
        optimizer_kwargs=dict(
            lr=0.007525067744232913
        ),
        regularizer='no'
    )
