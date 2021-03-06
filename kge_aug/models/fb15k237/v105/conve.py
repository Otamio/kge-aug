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
        model='ConvE',
        model_kwargs=dict(
            embedding_dim=256,
            feature_map_dropout=0.38074998430562207,
            input_dropout=0.481083618149555,
            output_channels=56,
            output_dropout=0.4920249242322924,
            # automatic_memory_optimization=True
        ),
        training_loop='LCWA',
        training_kwargs=dict(
            num_epochs=constants.epochs,
            batch_size=512,
            label_smoothing=0.05422578918650805
        ),
        loss="bceaftersigmoid",
        optimizer='Adam',
        optimizer_kwargs=dict(
            lr=0.0052417396207321025
        ),
        regularizer="no",
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
