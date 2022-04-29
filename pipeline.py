import tensorflow as tf
print('TensorFlow version: {}'.format(tf.__version__))
import tfx
print('TFX version: {}'.format(tfx.__version__))


def create_pipeline(pipeline_name: str, pipeline_root: str, data_root: str,
                     module_file: str, serving_model_dir: str,
                     metadata_path: str):

    # Pull In CSV dataset
    csv_component = tfx.components.CsvExampleGen(input_base=data_root)

    # Gather Dataset statistics
    statistics_gen = tfx.components.StatisticsGen(
        examples=csv_component.outputs['examples'],
        name='iris-stats'
    )

    # Analysis of the data schema
    infer_schema = tfx.components.SchemaGen(statistics=statistics_gen.outputs['statistics'])

    # Validate that the dataset is good for training
    validate_stats = tfx.components.ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=infer_schema.outputs['schema']
    )

    # Train data
    trainer = tfx.components.Trainer(
        module_file=module_file,
        examples=csv_component.outputs['examples'],
        train_args=tfx.proto.trainer_pb2.TrainArgs(num_steps=50),
        eval_args=tfx.proto.trainer_pb2.EvalArgs(num_steps=10))


    # Evaluates the model
    model_analyzer = tfx.components.Evaluator(
        examples=csv_component.outputs['examples'],
        model=trainer.outputs['model']
    )

    # Compare the model to prevous models
    model_validator = tfx.components.model_validator.component.ModelValidator(
        examples=csv_component.outputs['examples'],
        model=trainer.outputs['model']
    )

    # Push model to production
    pusher = tfx.components.Pusher(
        model=trainer.outputs['model'],
        push_destination=tfx.proto.pusher_pb2.PushDestination(
            filesystem=tfx.proto.pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir
            )
        )
    )

    components = [
        csv_component,
        statistics_gen,
        infer_schema,
        trainer,
        model_analyzer,
        pusher,
    ]

    return tfx.orchestration.pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(metadata_path),
        components=components
    )