from pykeen.pipeline import pipeline
from kge_aug.models import constants, get_data


def get_pipeline(dataset, training_fname):

    training, testing, validation = get_data.get(dataset, training_fname)

    return pipeline(
        training_triples_factory=training,
        testing_triples_factory=testing,
        validation_triples_factory=validation,
        dataset_kwargs=dict(
            create_inverse_triples=True
        ),
        model='TuckER',
        model_kwargs=dict(
            dropout_0=0.3516018143494416,
            dropout_1=0.47086223805162364,
            dropout_2=0.46634205793718553,
            embedding_dim=128,
            relation_dim=64,
            apply_batch_normalization=True,
            automatic_memory_optimization=True
        ),
        training_loop='LCWA',
        training_kwargs=dict(
            num_epochs=constants.epochs,
            batch_size=128,
            label_smoothing=0.009677960598649829
        ),
        loss="softplus",
        optimizer='Adam',
        optimizer_kwargs=dict(
            lr=0.0014058561550531544
        ),
        regularizer="no",
        evaluator="rankbased",
        evaluator_kwargs=dict(
            filtered=True
        ),
        stopper='early',
        stopper_kwargs=dict(
            metric=constants.metric,
            patience=constants.patience,
            delta=constants.delta
        )
    )
