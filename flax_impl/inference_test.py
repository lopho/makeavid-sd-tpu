from flax_trainer import FlaxTrainerUNetPseudo3D
import os
from flax import jax_utils

trainer = FlaxTrainerUNetPseudo3D(
        model_path = '../storage/trained_models/ep20',
        from_pt = False,
        convert2d = False,
        sample_size = (64, 64),
        seed = 0,
        dtype = 'bfloat16',
        param_dtype = 'float32',
        use_memory_efficient_attention = True,
        verbose = True,
        only_temporal = True
)

params = jax_utils.replicate(trainer.params)

def sample(
        prompt = 'dancing person',
        num_frames = 24,
        neg_prompt = '',
        steps = 50,
        cfg = 9.0,
        image_path = 'testimage.png'
) -> None:
    images = trainer.sample(
            params = params,
            prompt = prompt,
            num_frames = num_frames,
            replicate_params = False,
            neg_prompt = neg_prompt,
            steps = steps,
            cfg = cfg,
            image_path = image_path
    )
    os.makedirs('inference_out', exist_ok = True)
    for i, im in enumerate(images):
        im.save(os.path.join('inference_out', str(i).zfill(5) + '.png'), optimize = True)