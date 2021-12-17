from pykeen.pipeline import pipeline
from datasets import constants
from pykeen.triples import TriplesFactory

def get_pipeline():

    training = TriplesFactory(path=f"{constants.dataset_nations}/train.txt")
    testing = TriplesFactory(
        path=f"{constants.dataset_nations}/test.txt",
        entity_to_id=training.entity_to_id,
        relation_to_id=training.relation_to_id
    )
    validation = TriplesFactory(
        path=f"{constants.dataset_nations}/valid.txt",
        entity_to_id=training.entity_to_id,
        relation_to_id=training.relation_to_id
    )

    return pipeline(
        training_triples_factory=training,
        testing_triples_factory=testing,
        validation_triples_factory=validation,
        dataset_kwargs=dict(
            create_inverse_triples=True
        ),
        model='ConvE',
        model_kwargs=dict(
            embedding_dim=constants.dimension,
            input_channels=1,
            output_channels=32,
            embedding_height=10,
            embedding_width=20,
            kernel_height=3,
            kernel_width=3,
            input_dropout=0.2,
            feature_map_dropout=0.2,
            output_dropout=0.3,
            apply_batch_normalization=True,
            # automatic_memory_optimization=True
        ),
        training_loop='LCWA',
        # training_loop_kwargs=dict(
        #     automatic_memory_optimization=True
        # ),
        training_kwargs=dict(
            num_epochs=constants.epochs,
            batch_size=128,
            label_smoothing=0.1
        ),
        loss="BCEAfterSigmoidLoss",
        loss_kwargs=dict(
            reduction="mean"
        ),
        optimizer='Adam',
        optimizer_kwargs=dict(
            lr=0.0001
        ),
        evaluator_kwargs=dict(
            filtered=True
        ),
    )