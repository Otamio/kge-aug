from pykeen.pipeline import pipeline
from kge_aug.models import constants


def get_pipeline(training, testing, validation):

    return pipeline(
        training=training,
        testing=testing,
        validation=validation,
        dataset_kwargs=dict(
            create_inverse_triples=True
        ),
        model='ComplEx',
        model_kwargs=dict(
            embedding_dim=256
        ),
        training_loop='LCWA',
        training_kwargs=dict(
            num_epochs=constants.epochs,
            batch_size=256,
            label_smoothing=0.08094657004944494
        ),
        loss="crossentropy",
        optimizer='Adam',
        optimizer_kwargs=dict(
            lr=0.007525067744232913
        ),
        regularizer='no',
        evaluator_kwargs=dict(
            filtered=True
        ),
        stopper='early',
        stopper_kwargs=dict(
            metric=constants.metric,
            patience=constants.patience,
            relative_delta=constants.relative_delta
        )
    )
