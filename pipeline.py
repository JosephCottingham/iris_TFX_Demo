import tensorflow as tf
print('TensorFlow version: {}'.format(tf.__version__))
import tfx
print('TFX version: {}'.format(tfx.__version__))

import tensorflow_model_analysis as tfma

def create_pipeline(pipeline_name: str, pipeline_root: str, data_root: str,
                     module_file: str, serving_model_dir: str,
                     metadata_path: str):

    # Pull In CSV dataset
    csv_component = tfx.components.CsvExampleGen(input_base=data_root)

    # Gather Dataset statistics
    statistics_component = tfx.components.StatisticsGen(
        examples=csv_component.outputs['examples']
    )

    # Analysis of the data schema
    schema_component = tfx.components.SchemaGen(statistics=statistics_component.outputs['statistics'])

    # Validate that the dataset is good for training
    validate_stats = tfx.components.ExampleValidator(
        statistics=statistics_component.outputs['statistics'],
        schema=schema_component.outputs['schema']
    )

    # Train data
    trainer_component = tfx.components.Trainer(
        module_file=module_file,
        examples=csv_component.outputs['examples'],
        train_args=tfx.proto.trainer_pb2.TrainArgs(num_steps=50),
        eval_args=tfx.proto.trainer_pb2.EvalArgs(num_steps=10)
    )


    # Evaluates the model

    eval_config = tfma.EvalConfig(
    model_specs=[
        # This assumes a serving model with signature 'serving_default'. If
        # using estimator based EvalSavedModel, add signature_name='eval' and
        # remove the label_key. Note, if using a TFLite model, then you must set
        # model_type='tf_lite'.
        tfma.ModelSpec(label_key='<label_key>')
    ],
    metrics_specs=[
        tfma.MetricsSpec(
            # The metrics added here are in addition to those saved with the
            # model (assuming either a keras model or EvalSavedModel is used).
            # Any metrics added into the saved model (for example using
            # model.compile(..., metrics=[...]), etc) will be computed
            # automatically.
            metrics=[
                tfma.MetricConfig(class_name='ExampleCount'),
                tfma.MetricConfig(
                    class_name='BinaryAccuracy',
                    threshold=tfma.MetricThreshold(
                        value_threshold=tfma.GenericValueThreshold(
                            lower_bound={'value': 0.5}),
                        change_threshold=tfma.GenericChangeThreshold(
                            direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                            absolute={'value': -1e-10})))
            ]
        )
    ],
    slicing_specs=[
        # An empty slice spec means the overall slice, i.e. the whole dataset.
        tfma.SlicingSpec(),
        # Data can be sliced along a feature column. In this case, data is
        # sliced along feature column trip_start_hour.
        tfma.SlicingSpec(feature_keys=['trip_start_hour'])
    ])

    evaluator_component = tfx.components.Evaluator(
        examples=csv_component.outputs['examples'],
        model=trainer_component.outputs['model'],
        eval_config=eval_config
    )

    # Push model to production
    pusher_component = tfx.components.Pusher(
        model=trainer_component.outputs['model'],
        push_destination=tfx.proto.pusher_pb2.PushDestination(
            filesystem=tfx.proto.pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir
            )
        )
    )

    components = [
        csv_component,
        statistics_component,
        schema_component,
        trainer_component,
        evaluator_component,
        pusher_component,
    ]

    return tfx.orchestration.pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(metadata_path),
        components=components
    )