# Make-A-Video Latent Diffusion Models
# Copyright (C) 2023  Lopho <contact@lopho.org>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from typing import Any, Union, Optional, Tuple, List, Dict
import gc
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from flax.core.frozen_dict import FrozenDict
from flax import jax_utils
from flax.training.common_utils import shard
from PIL import Image
import einops

from diffusers import FlaxAutoencoderKL, FlaxUNet2DConditionModel
from diffusers import (
        FlaxDDIMScheduler,
        FlaxPNDMScheduler,
        FlaxLMSDiscreteScheduler,
        FlaxDPMSolverMultistepScheduler,
)
from diffusers.schedulers.scheduling_ddim_flax import DDIMSchedulerState
from diffusers.schedulers.scheduling_pndm_flax import PNDMSchedulerState
from diffusers.schedulers.scheduling_lms_discrete_flax import LMSDiscreteSchedulerState
from diffusers.schedulers.scheduling_dpmsolver_multistep_flax import DPMSolverMultistepSchedulerState

from transformers import FlaxCLIPTextModel, CLIPTokenizer

from makeavid_sd.flax_impl import FlaxUNetPseudo3DConditionModel

SchedulerType = Union[
        FlaxDDIMScheduler,
        FlaxPNDMScheduler,
        FlaxLMSDiscreteScheduler,
        FlaxDPMSolverMultistepScheduler,
]

SchedulerStateType = Union[
        DDIMSchedulerState,
        PNDMSchedulerState,
        LMSDiscreteSchedulerState,
        DPMSolverMultistepSchedulerState,
]

SCHEDULERS: Dict[str, SchedulerType] = {
        'dpm': FlaxDPMSolverMultistepScheduler, # husbando
        'ddim': FlaxDDIMScheduler,
        #'PLMS': FlaxPNDMScheduler, # its not correctly implemented in diffusers, output is bad, but at least it "works"
        #'LMS': FlaxLMSDiscreteScheduler, # borked
        #    image_latents, image_scheduler_state = scheduler.step(
        #    File "/mnt/work1/make_a_vid/makeavid-space/.venv/lib/python3.10/site-packages/diffusers/schedulers/scheduling_lms_discrete_flax.py", line 255, in step
        #    order = min(timestep + 1, order)
        #    jax._src.errors.ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected: Traced<ShapedArray(bool[])>with<DynamicJaxprTrace(level=1/1)>
        #    The problem arose with the `bool` function. 
        # The error occurred while tracing the function scanned_fun at /mnt/work1/make_a_vid/makeavid-space/.venv/lib/python3.10/site-packages/jax/_src/lax/control_flow/loops.py:1668 for scan. This concrete value was not available in Python because it depends on the values of the arguments loop_carry[0] and loop_carry[1][1].timesteps
}

def dtypestr(x: jnp.dtype):
    if x == jnp.float32: return 'float32'
    elif x == jnp.float16: return 'float16'
    elif x == jnp.bfloat16: return 'bfloat16'
    else: raise
def castto(dtype, m, x):
    if dtype == jnp.float32: return m.to_fp32(x)
    elif dtype == jnp.float16: return m.to_fp16(x)
    elif dtype == jnp.bfloat16: return m.to_bf16(x)
    else: raise

class InferenceUNetPseudo3D:
    def __init__(self,
            model_path: str,
            dtype: jnp.dtype = jnp.float16,
            hf_auth_token: Union[str, None] = None
    ) -> None:
        self.dtype = dtype
        self.model_path = model_path
        self.hf_auth_token = hf_auth_token

        self.params: Dict[str, FrozenDict[str, Any]] = {}
        try:
            import traceback
            print('initializing unet')
            unet, unet_params = FlaxUNetPseudo3DConditionModel.from_pretrained(
                    self.model_path,
                    subfolder = 'unet',
                    from_pt = False,
                    sample_size = (64, 64),
                    dtype = self.dtype,
                    param_dtype = dtypestr(self.dtype),
                    use_memory_efficient_attention = True,
                    use_auth_token = self.hf_auth_token
            )
            self.unet: FlaxUNetPseudo3DConditionModel = unet
            print('casting unet params')
            unet_params = castto(self.dtype, self.unet, unet_params)
            print('storing unet params')
            self.params['unet'] = FrozenDict(unet_params)
            print('deleting unet params')
            del unet_params
        except Exception as e:
            print(e)
            self.failed = ''.join(traceback.format_exception(None, e, e.__traceback__))
            traceback.print_exc()
            return
        self.failed = False
        vae, vae_params = FlaxAutoencoderKL.from_pretrained(
                self.model_path,
                subfolder = 'vae',
                from_pt = True,
                dtype = self.dtype,
                use_auth_token = self.hf_auth_token
        )
        self.vae: FlaxAutoencoderKL = vae
        vae_params = castto(self.dtype, self.vae, vae_params)
        self.params['vae'] = FrozenDict(vae_params)
        del vae_params
        text_encoder = FlaxCLIPTextModel.from_pretrained(
                self.model_path,
                subfolder = 'text_encoder',
                from_pt = True,
                dtype = self.dtype,
                use_auth_token = self.hf_auth_token
        )
        text_encoder_params = text_encoder.params
        del text_encoder._params
        text_encoder_params = castto(self.dtype, text_encoder, text_encoder_params)
        self.text_encoder: FlaxCLIPTextModel = text_encoder
        self.params['text_encoder'] = FrozenDict(text_encoder_params)
        del text_encoder_params
        imunet, imunet_params = FlaxUNet2DConditionModel.from_pretrained(
                'runwayml/stable-diffusion-v1-5',
                subfolder = 'unet',
                from_pt = True,
                dtype = self.dtype,
                use_memory_efficient_attention = True,
                use_auth_token = self.hf_auth_token
        )
        imunet_params = castto(self.dtype, imunet, imunet_params)
        self.imunet: FlaxUNet2DConditionModel = imunet
        self.params['imunet'] = FrozenDict(imunet_params)
        del imunet_params
        self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(
                self.model_path,
                subfolder = 'tokenizer',
                use_auth_token = self.hf_auth_token
        )
        self.schedulers: Dict[str, Dict[str, SchedulerType]] = {}
        for scheduler_name in SCHEDULERS:
            if scheduler_name not in ['KarrasVe', 'SDEVe']:
                scheduler, scheduler_state = SCHEDULERS[scheduler_name].from_pretrained(
                        self.model_path,
                        subfolder = 'scheduler',
                        dtype = jnp.float32,
                        use_auth_token = self.hf_auth_token
                )
            else:
                scheduler, scheduler_state = SCHEDULERS[scheduler_name].from_pretrained(
                        self.model_path,
                        subfolder = 'scheduler',
                        use_auth_token = self.hf_auth_token
                )
            self.schedulers[scheduler_name] = scheduler
            self.params[scheduler_name] = scheduler_state
        self.vae_scale_factor: int = int(2 ** (len(self.vae.config.block_out_channels) - 1))
        self.device_count = jax.device_count()
        gc.collect()

    def prepare_inputs(self,
            prompt: List[str],
            neg_prompt: List[str],
            hint_image: List[Image.Image],
            mask_image: List[Image.Image],
            width: int,
            height: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]: # prompt, neg_prompt, hint_image, mask_image
        tokens = self.tokenizer(
            prompt,
            truncation = True,
            return_overflowing_tokens = False,
            max_length = 77, #self.text_encoder.config.max_length defaults to 20 if its not in the config smh
            padding = 'max_length',
            return_tensors = 'np'
        ).input_ids
        tokens = jnp.array(tokens, dtype = jnp.int32)
        neg_tokens = self.tokenizer(
            neg_prompt,
            truncation = True,
            return_overflowing_tokens = False,
            max_length = 77,
            padding = 'max_length',
            return_tensors = 'np'
        ).input_ids
        neg_tokens = jnp.array(neg_tokens, dtype = jnp.int32)
        for i,im in enumerate(hint_image):
            if im.size != (width, height):
                hint_image[i] = hint_image[i].resize((width, height), resample = Image.Resampling.LANCZOS)
        for i,im in enumerate(mask_image):
            if im.size != (width, height):
                mask_image[i] = mask_image[i].resize((width, height), resample = Image.Resampling.LANCZOS)
        # b,h,w,c | c == 3
        hint = jnp.concatenate(
                [ jnp.expand_dims(np.asarray(x.convert('RGB')), axis = 0) for x in hint_image ],
                axis = 0
        ).astype(jnp.float32)
        # scale -1,1
        hint = (hint / 255) * 2 - 1
        # b,h,w,c | c == 1
        mask = jnp.concatenate(
                [ jnp.expand_dims(np.asarray(x.convert('L')), axis = (0, -1)) for x in mask_image ],
                axis = 0
        ).astype(jnp.float32)
        # scale -1,1
        mask = (mask / 255) * 2 - 1
        # binarize mask
        mask = mask.at[mask < 0.5].set(0)
        mask = mask.at[mask >= 0.5].set(1)
        # mask hint image
        hint = hint * (mask < 0.5)
        # b,h,w,c -> b,c,h,w
        hint = hint.transpose((0,3,1,2))
        mask = mask.transpose((0,3,1,2))
        return tokens, neg_tokens, hint, mask

    def generate(self,
            prompt: Union[str, List[str]] = '',
            inference_steps: int = 20,
            hint_image: Union[Image.Image, List[Image.Image], None] = None,
            mask_image: Union[Image.Image, List[Image.Image], None] = None,
            neg_prompt: Union[str, List[str]] = '',
            cfg: float = 15.0,
            cfg_image: Optional[float] = None,
            num_frames: int = 24,
            width: int = 512,
            height: int = 512,
            seed: int = 0,
            scheduler_type: str = 'dpm'
    ) -> List[List[Image.Image]]:
        assert inference_steps > 0, f'number of inference steps must be > 0 but is {inference_steps}'
        assert num_frames > 0, f'number of frames must be > 0 but is {num_frames}'
        assert width % 32 == 0, f'width must be divisible by 32 but is {width}'
        assert height % 32 == 0, f'height must be divisible by 32 but is {height}'
        if isinstance(prompt, str):
            prompt = [ prompt ]
        batch_size = len(prompt)
        assert batch_size % self.device_count == 0, f'batch size must be multiple of {self.device_count}'
        if hint_image is None:
            hint_image = Image.new('RGB', (width, height), color = (0,0,0))
            use_imagegen = True
        else:
            use_imagegen = False
        if isinstance(hint_image, Image.Image):
            hint_image = [ hint_image ] * batch_size
        assert len(hint_image) == batch_size, f'number of hint images must be equal to batch size {batch_size} but is {len(hint_image)}'
        if mask_image is None:
            mask_image = Image.new('L', hint_image[0].size, color = 0)
        if isinstance(mask_image, Image.Image):
            mask_image = [ mask_image ] * batch_size
        assert len(mask_image) == batch_size, f'number of mask images must be equal to batch size {batch_size} but is {len(mask_image)}'
        if isinstance(neg_prompt, str):
            neg_prompt = [ neg_prompt ] * batch_size
        assert len(neg_prompt) == batch_size, f'number of negative prompts must be equal to batch size {batch_size} but is {len(neg_prompt)}'
        assert scheduler_type in SCHEDULERS, f'unknown type of noise scheduler: {scheduler_type}, must be one of {list(SCHEDULERS.keys())}'
        tokens, neg_tokens, hint, mask = self.prepare_inputs(
                prompt = prompt,
                neg_prompt = neg_prompt,
                hint_image = hint_image,
                mask_image = mask_image,
                width = width,
                height = height
        )
        if cfg_image is None:
            cfg_image = cfg
        #params['scheduler'] = scheduler_state
        # NOTE splitting rngs is not deterministic,
        # running on different device counts gives different seeds
        #rng = jax.random.PRNGKey(seed)
        #rngs = jax.random.split(rng, self.device_count)
        # manually assign seeded RNGs to devices for reproducability
        rngs = jnp.array([ jax.random.PRNGKey(seed + i) for i in range(self.device_count) ])
        params = jax_utils.replicate(self.params)
        tokens = shard(tokens)
        neg_tokens = shard(neg_tokens)
        hint = shard(hint)
        mask = shard(mask)
        images = _p_generate(self,
            tokens,
            neg_tokens,
            hint,
            mask,
            inference_steps,
            num_frames,
            height,
            width,
            cfg,
            cfg_image,
            rngs,
            params,
            use_imagegen,
            scheduler_type,
        )
        if images.ndim == 5:
            images = einops.rearrange(images, 'd f c h w -> (d f) h w c')
        else:
            images = einops.rearrange(images, 'f c h w -> f h w c')
        # to cpu
        images = np.array(images)
        images = [ Image.fromarray(x) for x in images ]
        return images

    def _generate(self,
            tokens: jnp.ndarray,
            neg_tokens: jnp.ndarray,
            hint: jnp.ndarray,
            mask: jnp.ndarray,
            inference_steps: int,
            num_frames,
            height,
            width,
            cfg: float,
            cfg_image: float,
            rng: jax.random.KeyArray,
            params: Union[Dict[str, Any], FrozenDict[str, Any]],
            use_imagegen: bool,
            scheduler_type: str
    ) -> List[Image.Image]:
        batch_size = tokens.shape[0]
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        latent_shape = (
                batch_size,
                self.vae.config.latent_channels,
                num_frames,
                latent_h,
                latent_w
        )
        encoded_prompt = self.text_encoder(tokens, params = params['text_encoder'])[0]
        encoded_neg_prompt = self.text_encoder(neg_tokens, params = params['text_encoder'])[0]

        scheduler = self.schedulers[scheduler_type]
        scheduler_state = params[scheduler_type]

        if use_imagegen:
            image_latent_shape = (batch_size, self.vae.config.latent_channels, latent_h, latent_w)
            image_latents = jax.random.normal(
                    rng,
                    shape = image_latent_shape,
                    dtype = jnp.float32
            ) * scheduler_state.init_noise_sigma
            image_scheduler_state = scheduler.set_timesteps(
                    scheduler_state,
                    num_inference_steps = inference_steps,
                    shape = image_latents.shape
            )
            def image_sample_loop(step, args):
                image_latents, image_scheduler_state = args
                t = image_scheduler_state.timesteps[step]
                tt = jnp.broadcast_to(t, image_latents.shape[0])
                latents_input = scheduler.scale_model_input(image_scheduler_state, image_latents, t)
                noise_pred = self.imunet.apply(
                        { 'params': params['imunet']} ,
                        latents_input,
                        tt,
                        encoder_hidden_states = encoded_prompt
                ).sample
                noise_pred_uncond = self.imunet.apply(
                        { 'params': params['imunet'] },
                        latents_input,
                        tt,
                        encoder_hidden_states = encoded_neg_prompt
                ).sample
                noise_pred = noise_pred_uncond + cfg_image * (noise_pred - noise_pred_uncond)
                image_latents, image_scheduler_state = scheduler.step(
                        image_scheduler_state,
                        noise_pred.astype(jnp.float32),
                        t,
                        image_latents
                ).to_tuple()
                return image_latents, image_scheduler_state
            image_latents, _ = jax.lax.fori_loop(
                    0, inference_steps,
                    image_sample_loop,
                    (image_latents, image_scheduler_state)
            )
            hint = image_latents
        else:
            hint = self.vae.apply(
                    { 'params': params['vae'] },
                    hint,
                    method = self.vae.encode
            ).latent_dist.mean * self.vae.config.scaling_factor
            # NOTE vae keeps channels last for encode, but rearranges to channels first for decode
            # b0 h1 w2 c3 -> b0 c3 h1 w2
            hint = hint.transpose((0, 3, 1, 2))

        hint = jnp.expand_dims(hint, axis = 2).repeat(num_frames, axis = 2)
        mask = jax.image.resize(mask, (*mask.shape[:-2], *hint.shape[-2:]), method = 'nearest')
        mask = jnp.expand_dims(mask, axis = 2).repeat(num_frames, axis = 2)
        # NOTE jax normal distribution is shit with float16 + bfloat16
        # SEE https://github.com/google/jax/discussions/13798
        # generate random at float32
        latents = jax.random.normal(
                rng,
                shape = latent_shape,
                dtype = jnp.float32
        ) * scheduler_state.init_noise_sigma
        scheduler_state = scheduler.set_timesteps(
                scheduler_state,
                num_inference_steps = inference_steps,
                shape = latents.shape
        )

        def sample_loop(step, args):
            latents, scheduler_state = args
            t = scheduler_state.timesteps[step]#jnp.array(scheduler_state.timesteps, dtype = jnp.int32)[step]
            tt = jnp.broadcast_to(t, latents.shape[0])
            latents_input = scheduler.scale_model_input(scheduler_state, latents, t)
            latents_input = jnp.concatenate([latents_input, mask, hint], axis = 1)
            noise_pred = self.unet.apply(
                    { 'params': params['unet'] },
                    latents_input,
                    tt,
                    encoded_prompt
            ).sample
            noise_pred_uncond = self.unet.apply(
                    { 'params': params['unet'] },
                    latents_input,
                    tt,
                    encoded_neg_prompt
            ).sample
            noise_pred = noise_pred_uncond + cfg * (noise_pred - noise_pred_uncond)
            latents, scheduler_state = scheduler.step(
                    scheduler_state,
                    noise_pred.astype(jnp.float32),
                    t,
                    latents
            ).to_tuple()
            return latents, scheduler_state

        latents, _ = jax.lax.fori_loop(
                0, inference_steps,
                sample_loop,
                (latents, scheduler_state)
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        latents = einops.rearrange(latents, 'b c f h w -> (b f) c h w')
        num_images = len(latents)
        images_out = jnp.zeros(
                (
                        num_images,
                        self.vae.config.out_channels,
                        height,
                        width
                ),
                dtype = self.dtype
        )
        def decode_loop(step, images_out):
            # NOTE vae keeps channels last for encode, but rearranges to channels first for decode
            im = self.vae.apply(
                    { 'params': params['vae'] },
                    jnp.expand_dims(latents[step], axis = 0),
                    method = self.vae.decode
            ).sample
            images_out = images_out.at[step].set(im[0])
            return images_out
        images_out = jax.lax.fori_loop(0, num_images, decode_loop, images_out)
        images_out = ((images_out / 2 + 0.5) * 255).round().clip(0, 255).astype(jnp.uint8)
        return images_out


@partial(
        jax.pmap,
        in_axes = ( # 0 -> split across batch dim, None -> duplicate
                None,   #  0 inference_class
                0,      #  1 tokens
                0,      #  2 neg_tokens
                0,      #  3 hint
                0,      #  4 mask
                None,   #  5 inference_steps
                None,   #  6 num_frames
                None,   #  7 height
                None,   #  8 width
                None,   #  9 cfg
                None,   # 10 cfg_image
                0,      # 11 rng
                0,      # 12 params
                None,   # 13 use_imagegen
                None,   # 14 scheduler_type
        ),
        static_broadcasted_argnums = ( # trigger recompilation (if cache miss) on change
                0,      # inference_class
                5,      # inference_steps
                6,      # num_frames
                7,      # height
                8,      # width
                13,     # use_imagegen
                14,     # scheduler_type
        )
)
def _p_generate(
        inference_class: InferenceUNetPseudo3D,
        tokens,
        neg_tokens,
        hint,
        mask,
        inference_steps: int,
        num_frames: int,
        height: int,
        width: int,
        cfg: float,
        cfg_image: float,
        rng,
        params,
        use_imagegen: bool,
        scheduler_type: str
):
    return inference_class._generate(
            tokens,
            neg_tokens,
            hint,
            mask,
            inference_steps,
            num_frames,
            height,
            width,
            cfg,
            cfg_image,
            rng,
            params,
            use_imagegen,
            scheduler_type
    )

