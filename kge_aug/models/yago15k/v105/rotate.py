from pykeen.pipeline import pipeline
from kge_aug.models import constants


def get_pipeline(training, testing, validation):

    return pipeline(
        training=training,
        testing=testing,
        validation=validation,
        dataset_kwargs=dict(
            create_inverse_triples=False
        ),
        model='RotatE',
        model_kwargs=dict(
            embedding_dim=128
        ),
        training_loop='sLCWA',
        training_kwargs=dict(
            num_epochs=constants.epochs,
            batch_size=512,
            label_smoothing=0.0
        ),
        loss="nssa",
        loss_kwargs=dict(
            adversarial_temperature=0.9053291478133825,
            margin=27.868096306273824
        ),
        optimizer='Adam',
        optimizer_kwargs=dict(
            lr=0.002750349821235516
        ),
        negative_sampler='basic',
        negative_sampler_kwargs=dict(
            num_negs_per_pos=79
        ),
        regularizer='no',
        evaluator="rankbased",
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
