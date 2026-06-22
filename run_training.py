import argparse
import inspect

from models import MDSelfS2Model
from trainers.base_config import BaseConfig
from trainers.self_trainer import SelfTrainer


def _base(model_class, epochs, lr=0.0001, batch_size=2, input_bands='10m', output_bands='10m', aux_frames=4,
          model_params=None, training_aux_strategy='random', randomness_closest_dates=0.0,
          training_orbit_source='mixed',
          use_alignment=False, resume_checkpoint_date=None, compile_model=False, mixed_precision='no'):
    frame = inspect.currentframe()
    exp_name = frame.f_back.f_code.co_name.removesuffix("_resume")

    if model_params is None:
        model_params = dict(
            input_bands=input_bands,
            output_bands=output_bands,
            multi_pred=[0] if aux_frames > 0 else None,
        )

    return BaseConfig(
        exp_name=exp_name,
        model_class=model_class,

        dataset_train_params=dict(
            data_type='train',
            normalize=256,
            aux_frames=aux_frames,
            aux_selection_strategy=training_aux_strategy,
            aux_fill='random',
            aux_source='both',
            aux_orbit_source=training_orbit_source,
            randomness_closest_dates=randomness_closest_dates,
            limit_samples=200,
        ),

        dataset_val_params=dict(
            data_type='val',
            normalize=256,
            aux_frames=aux_frames,
            aux_selection_strategy='random_fixed',
            aux_fill='mirror',
            aux_source='both',
            aux_orbit_source='mixed',
            limit_samples=20,
        ),

        input_bands=input_bands,
        output_bands=output_bands,
        use_alignment=use_alignment,
        model_params=model_params,
        loss_expr_20m="accurate",
        loss_expr_10m="accurate",

        epochs=epochs,

        lr=lr,
        schedule_warmup_epochs=0,
        mixed_precision=mixed_precision,
        batch_accum=1,
        save_checkpoint_every=1,
        save_images_len=100 // batch_size,
        save_images_every=2,
        num_workers=0,
        batch_size=batch_size,
        augmentation=True,
        compile_model=compile_model,

        print_step_loss=False,
        resume_checkpoint_date=resume_checkpoint_date,
    )


def sdselfsen2():
    return _base(MDSelfS2Model, 100, aux_frames=0, input_bands='10m_20m', output_bands='10m_20m',
                 mixed_precision='fp16')

def sdselfsen2_only10m():
    return _base(MDSelfS2Model, 100, aux_frames=0, input_bands='10m', output_bands='10m',
                 mixed_precision='fp16')

def mdselfsen2():
    return _base(MDSelfS2Model, 200, aux_frames=3, input_bands='10m_20m', output_bands='10m_20m',
                 training_aux_strategy='closest_dates', randomness_closest_dates=0.4, use_alignment=False,
                 mixed_precision='fp16')

def mdselfsen2_only10m():
    return _base(MDSelfS2Model, 200, aux_frames=3, input_bands='10m', output_bands='10m',
                 training_aux_strategy='closest_dates', randomness_closest_dates=0.4, use_alignment=False,
                 mixed_precision='fp16')



def main():
    parser = argparse.ArgumentParser(description="Run training script")
    parser.add_argument("experiment", type=str, help="Experiment name to run, e.g. sdselfsen2 or mdselfs2")

    args = parser.parse_args()
    config = globals()[args.experiment]()

    trainer = SelfTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
