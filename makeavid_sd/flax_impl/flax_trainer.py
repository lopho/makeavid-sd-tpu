
from typing import Any, Optional, Union, Tuple, Dict, List

import os
import random
import math
import time
import numpy as np
from tqdm.auto import tqdm, trange

import torch
from torch.utils.data import DataLoader

import jax
import jax.numpy as jnp
import optax
from flax import jax_utils, traverse_util
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState
from flax.training.common_utils import shard

# convert 2D -> 3D
from diffusers import FlaxUNet2DConditionModel

# inference test, run on these on cpu
from diffusers import AutoencoderKL
from diffusers.schedulers.scheduling_ddim_flax import FlaxDDIMScheduler, DDIMSchedulerState
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image


from .flax_unet_pseudo3d_condition import UNetPseudo3DConditionModel


def seed_all(seed: int) -> jax.random.PRNGKeyArray:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = jax.random.PRNGKey(seed)
    return rng

def count_params(
        params: Union[Dict[str, Any],
        FrozenDict[str, Any]],
        filter_name: Optional[str] = None
) -> int:
    p: Dict[Tuple[str], jax.Array] = traverse_util.flatten_dict(params)
    cc = 0
    for k in p:
        if filter_name is not None:
            if filter_name in ' '.join(k):
                cc += len(p[k].flatten())
        else:
            cc += len(p[k].flatten())
    return cc

def map_2d_to_pseudo3d(
        params2d: Dict[str, Any],
        params3d: Dict[str, Any],
        verbose: bool = True
) -> Dict[str, Any]:
    params2d = traverse_util.flatten_dict(params2d)
    params3d = traverse_util.flatten_dict(params3d)
    new_params = dict()
    for k in params3d:
        if 'spatial_conv' in k:
            k2d = list(k)
            k2d.remove('spatial_conv')
            k2d = tuple(k2d)
            if verbose:
                tqdm.write(f'Spatial: {k} <- {k2d}')
            p = params2d[k2d]
        elif k not in params2d:
            if verbose:
                tqdm.write(f'Missing: {k}')
            p = params3d[k]
        else:
            p = params2d[k]
        assert p.shape == params3d[k].shape, f'shape mismatch: {k}: {p.shape} != {params3d[k].shape}'
        new_params[k] = p
    new_params = traverse_util.unflatten_dict(new_params)
    return new_params


class FlaxTrainerUNetPseudo3D:
    def __init__(self,
            model_path: str,
            from_pt: bool = True,
            convert2d: bool = False,
            sample_size: Tuple[int, int] = (64, 64),
            seed: int = 0,
            dtype: str = 'float32',
            param_dtype: str = 'float32',
            only_temporal: bool = True,
            use_memory_efficient_attention = False,
            verbose: bool = True
    ) -> None:
        self.verbose = verbose
        self.tracker: Optional['wandb.sdk.wandb_run.Run'] = None
        self._use_wandb: bool = False
        self._tracker_meta: Dict[str, Union[float, int]] = {
            't00': 0.0,
            't0': 0.0,
            'step0': 0
        }

        self.log('Init JAX')
        self.num_devices = jax.device_count()
        self.log(f'Device count: {self.num_devices}')

        self.seed = seed
        self.rng: jax.random.PRNGKeyArray = seed_all(self.seed)

        self.sample_size = sample_size
        if dtype == 'float32':
            self.dtype = jnp.float32
        elif dtype == 'bfloat16':
            self.dtype = jnp.bfloat16
        elif dtype == 'float16':
            self.dtype = jnp.float16
        else:
            raise ValueError(f'unknown type: {dtype}')
        self.dtype_str: str = dtype
        if param_dtype not in ['float32', 'bfloat16', 'float16']:
            raise ValueError(f'unknown parameter type: {param_dtype}')
        self.param_dtype = param_dtype
        self._load_models(
                model_path = model_path,
                convert2d = convert2d,
                from_pt = from_pt,
                use_memory_efficient_attention = use_memory_efficient_attention
        )
        self._mark_parameters(only_temporal = only_temporal)
        # optionally for validation + sampling
        self.tokenizer: Optional[CLIPTokenizer] = None
        self.text_encoder: Optional[CLIPTextModel] = None
        self.vae: Optional[AutoencoderKL] = None
        self.ddim: Optional[Tuple[FlaxDDIMScheduler, DDIMSchedulerState]] = None

    def log(self, message: Any) -> None:
        if self.verbose and jax.process_index() == 0:
            tqdm.write(str(message))

    def log_metrics(self, metrics: dict, step: int, epoch: int) -> None:
        if jax.process_index() > 0 or (not self.verbose and self.tracker is None):
            return
        now = time.monotonic()
        log_data = {
                'train/step': step,
                'train/epoch': epoch,
                'train/steps_per_sec': (step - self._tracker_meta['step0']) / (now - self._tracker_meta['t0']),
                **{ f'train/{k}': v for k, v in metrics.items() }
        }
        self._tracker_meta['t0'] = now
        self._tracker_meta['step0'] = step
        self.log(log_data)
        if self.tracker is not None:
            self.tracker.log(log_data, step = step)


    def enable_wandb(self, enable: bool = True) -> None:
        self._use_wandb = enable

    def _setup_wandb(self, config: Dict[str, Any] = dict()) -> None:
        import wandb
        import wandb.sdk
        self.tracker: wandb.sdk.wandb_run.Run = wandb.init(
                config = config,
                settings = wandb.sdk.Settings(
                        username = 'anon',
                        host = 'anon',
                        email = 'anon',
                        root_dir = 'anon',
                        _executable = 'anon',
                        _disable_stats = True,
                        _disable_meta = True,
                        disable_code = True,
                        disable_git = True
                ) # pls don't log sensitive data like system user names. also, fuck you for even trying.
        )

    def _init_tracker_meta(self) -> None:
        now = time.monotonic()
        self._tracker_meta = {
            't00': now,
            't0': now,
            'step0': 0
        }

    def _load_models(self,
            model_path: str,
            convert2d: bool,
            from_pt: bool,
            use_memory_efficient_attention: bool
    ) -> None:
        self.log(f'Load pretrained from {model_path}')
        if convert2d:
            self.log('  Convert 2D model to Pseudo3D')
            self.log('    Initiate Pseudo3D model')
            config = UNetPseudo3DConditionModel.load_config(model_path, subfolder = 'unet')
            model = UNetPseudo3DConditionModel.from_config(
                    config,
                    sample_size = self.sample_size,
                    dtype = self.dtype,
                    param_dtype = self.param_dtype,
                    use_memory_efficient_attention = use_memory_efficient_attention
            )
            params: Dict[str, Any] = model.init_weights(self.rng).unfreeze()
            self.log('    Load 2D model')
            model2d, params2d = FlaxUNet2DConditionModel.from_pretrained(
                    model_path,
                    subfolder = 'unet',
                    dtype = self.dtype,
                    from_pt = from_pt
            )
            self.log('    Map 2D -> 3D')
            params = map_2d_to_pseudo3d(params2d, params, verbose = self.verbose)
            del params2d
            del model2d
            del config
        else:
            model, params = UNetPseudo3DConditionModel.from_pretrained(
                    model_path,
                    subfolder = 'unet',
                    from_pt = from_pt,
                    sample_size = self.sample_size,
                    dtype = self.dtype,
                    param_dtype = self.param_dtype,
                    use_memory_efficient_attention = use_memory_efficient_attention
            )
        self.log(f'Cast parameters to {model.param_dtype}')
        if model.param_dtype == 'float32':
            params = model.to_fp32(params)
        elif model.param_dtype == 'float16':
            params = model.to_fp16(params)
        elif model.param_dtype == 'bfloat16':
            params = model.to_bf16(params)
        self.pretrained_model = model_path
        self.model: UNetPseudo3DConditionModel = model
        self.params: FrozenDict[str, Any] = FrozenDict(params)

    def _mark_parameters(self, only_temporal: bool) -> None:
        self.log('Mark training parameters')
        if only_temporal:
            self.log('Only training temporal layers')
        if only_temporal:
            param_partitions = traverse_util.path_aware_map(
                    lambda path, _: 'trainable' if 'temporal' in ' '.join(path) else 'frozen', self.params
            )
        else:
            param_partitions = traverse_util.path_aware_map(
                    lambda *_: 'trainable', self.params
            )
        self.only_temporal = only_temporal
        self.param_partitions: FrozenDict[str, Any] = FrozenDict(param_partitions)
        self.log(f'Total parameters: {count_params(self.params)}')
        self.log(f'Temporal parameters: {count_params(self.params, "temporal")}')

    def _load_inference_models(self) -> None:
        assert jax.process_index() == 0, 'not main process'
        if self.text_encoder is None:
            self.log('Load text encoder')
            self.text_encoder = CLIPTextModel.from_pretrained(
                    self.pretrained_model,
                    subfolder = 'text_encoder'
            )
        if self.tokenizer is None:
            self.log('Load tokenizer')
            self.tokenizer = CLIPTokenizer.from_pretrained(
                    self.pretrained_model,
                    subfolder = 'tokenizer'
            )
        if self.vae is None:
            self.log('Load vae')
            self.vae = AutoencoderKL.from_pretrained(
                    self.pretrained_model,
                    subfolder = 'vae'
            )
        if self.ddim is None:
            self.log('Load ddim scheduler')
            # tuple(scheduler , scheduler state)
            self.ddim = FlaxDDIMScheduler.from_pretrained(
                    self.pretrained_model,
                    subfolder = 'scheduler',
                    from_pt = True
            )

    def _unload_inference_models(self) -> None:
        self.text_encoder = None
        self.tokenizer = None
        self.vae = None
        self.ddim = None

    def sample(self,
            params: Union[Dict[str, Any], FrozenDict[str, Any]],
            prompt: str,
            image_path: str,
            num_frames: int,
            replicate_params: bool = True,
            neg_prompt: str = '',
            steps: int = 50,
            cfg: float = 9.0,
            unload_after_usage: bool = False
    ) -> List[Image.Image]:
        assert jax.process_index() == 0, 'not main process'
        self.log('Sample')
        self._load_inference_models()
        with torch.no_grad():
            tokens = self.tokenizer(
                [ prompt ],
                truncation = True,
                return_overflowing_tokens = False,
                padding = 'max_length',
                return_tensors = 'pt'
            ).input_ids
            neg_tokens = self.tokenizer(
                [ neg_prompt ],
                truncation = True,
                return_overflowing_tokens = False,
                padding = 'max_length',
                return_tensors = 'pt'
            ).input_ids
            encoded_prompt = self.text_encoder(input_ids = tokens).last_hidden_state
            encoded_neg_prompt = self.text_encoder(input_ids = neg_tokens).last_hidden_state
            hint_latent = torch.tensor(np.asarray(Image.open(image_path))).permute(2,0,1).to(torch.float32).div(255).mul(2).sub(1).unsqueeze(0)
            hint_latent = self.vae.encode(hint_latent).latent_dist.mean * self.vae.config.scaling_factor  #0.18215 # deterministic
            hint_latent = hint_latent.unsqueeze(2).repeat_interleave(num_frames, 2)
            mask = torch.zeros_like(hint_latent[:,0:1,:,:,:]) # zero mask, e.g. skip masking for now
            init_latent = torch.randn_like(hint_latent)
            # move to devices
            encoded_prompt = jnp.array(encoded_prompt.numpy())
            encoded_neg_prompt = jnp.array(encoded_neg_prompt.numpy())
            hint_latent = jnp.array(hint_latent.numpy())
            mask = jnp.array(mask.numpy())
            init_latent = init_latent.repeat(jax.device_count(), 1, 1, 1, 1)
            init_latent = jnp.array(init_latent.numpy())
            self.ddim = (self.ddim[0], self.ddim[0].set_timesteps(self.ddim[1], steps))
            timesteps = self.ddim[1].timesteps
            if replicate_params:
                params = jax_utils.replicate(params)
            ddim_state = jax_utils.replicate(self.ddim[1])
            encoded_prompt = jax_utils.replicate(encoded_prompt)
            encoded_neg_prompt = jax_utils.replicate(encoded_neg_prompt)
            hint_latent = jax_utils.replicate(hint_latent)
            mask = jax_utils.replicate(mask)
            # sampling fun
            def sample_loop(init_latent, ddim_state, t, params, encoded_prompt, encoded_neg_prompt, hint_latent, mask):
                latent_model_input = jnp.concatenate([init_latent, mask, hint_latent], axis = 1)
                pred = self.model.apply(
                        { 'params': params },
                        latent_model_input,
                        t,
                        encoded_prompt
                ).sample
                if cfg != 1.0:
                    neg_pred = self.model.apply(
                            { 'params': params },
                            latent_model_input,
                            t,
                            encoded_neg_prompt
                    ).sample
                    pred = neg_pred + cfg * (pred - neg_pred)
                # TODO check if noise is added at the right dimension
                init_latent, ddim_state = self.ddim[0].step(ddim_state, pred, t, init_latent).to_tuple()
                return init_latent, ddim_state
            p_sample_loop = jax.pmap(sample_loop, 'sample', donate_argnums = ())
            pbar_sample = trange(len(timesteps), desc = 'Sample', dynamic_ncols = True, smoothing = 0.1, disable = not self.verbose)
            init_latent = shard(init_latent)
            for i in pbar_sample:
                t = timesteps[i].repeat(self.num_devices)
                t = shard(t)
                init_latent, ddim_state = p_sample_loop(init_latent, ddim_state, t, params, encoded_prompt, encoded_neg_prompt, hint_latent, mask)
            # decode
            self.log('Decode')
            init_latent = torch.tensor(np.array(init_latent))
            init_latent = init_latent / self.vae.config.scaling_factor
            # d:0 b:1 c:2 f:3 h:4 w:5 -> d b f c h w
            init_latent = init_latent.permute(0, 1, 3, 2, 4, 5)
            images = []
            pbar_decode = trange(len(init_latent), desc = 'Decode', dynamic_ncols = True)
            for sample in init_latent:
                ims = self.vae.decode(sample.squeeze()).sample
                ims = ims.add(1).div(2).mul(255).round().clamp(0, 255).to(torch.uint8).permute(0,2,3,1).numpy()
                ims = [ Image.fromarray(x) for x in ims ]
                for im in ims:
                    images.append(im)
                pbar_decode.update(1)
        if unload_after_usage:
            self._unload_inference_models()
        return images

    def get_params_from_state(self, state: TrainState) -> FrozenDict[Any, str]:
        return FrozenDict(jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state.params)))

    def train(self,
            dataloader: DataLoader,
            lr: float,
            num_frames: int,
            log_every_step: int = 10,
            save_every_epoch: int = 1,
            sample_every_epoch: int = 1,
            output_dir: str = 'output',
            warmup: float = 0,
            decay: float = 0,
            epochs: int = 10,
            weight_decay: float = 1e-2
    ) -> None:
        eps = 1e-8
        total_steps = len(dataloader) * epochs
        warmup_steps = math.ceil(warmup * total_steps) if warmup > 0 else 0
        decay_steps = math.ceil(decay * total_steps) + warmup_steps if decay > 0 else warmup_steps + 1
        self.log(f'Total steps:  {total_steps}')
        self.log(f'Warmup steps: {warmup_steps}')
        self.log(f'Decay steps:  {decay_steps - warmup_steps}')
        if warmup > 0 or decay > 0:
            if not decay > 0:
                # only warmup, keep peak lr until end
                self.log('Warmup schedule')
                end_lr = lr
            else:
                # warmup + annealing to end lr
                self.log('Warmup + cosine annealing schedule')
                end_lr = eps
            lr_schedule = optax.warmup_cosine_decay_schedule(
                    init_value = 0.0,
                    peak_value = lr,
                    warmup_steps = warmup_steps,
                    decay_steps = decay_steps,
                    end_value = end_lr
            )
        else:
            # no warmup or decay -> constant lr
            self.log('constant schedule')
            lr_schedule = optax.constant_schedule(value = lr)
        adamw = optax.adamw(
                learning_rate = lr_schedule,
                b1 = 0.9,
                b2 = 0.999,
                eps = eps,
                weight_decay = weight_decay #0.01 # 0.0001
        )
        optim = optax.chain(
                optax.clip_by_global_norm(max_norm = 1.0),
                adamw
        )
        partition_optimizers = {
                'trainable': optim,
                'frozen': optax.set_to_zero()
        }
        tx = optax.multi_transform(partition_optimizers, self.param_partitions)
        state = TrainState.create(
                apply_fn = self.model.__call__,
                params = self.params,
                tx = tx
        )
        validation_rng, train_rngs = jax.random.split(self.rng)
        train_rngs = jax.random.split(train_rngs, jax.local_device_count())

        def train_step(state: TrainState, batch: Dict[str, jax.Array], train_rng: jax.random.PRNGKeyArray):
            def compute_loss(
                    params: Dict[str, Any],
                    batch: Dict[str, jax.Array],
                    sample_rng: jax.random.PRNGKeyArray # unused, dataloader provides everything
            ) -> jax.Array:
                # 'latent_model_input': latent_model_input
                # 'encoder_hidden_states': encoder_hidden_states
                # 'timesteps': timesteps
                # 'noise': noise
                latent_model_input = batch['latent_model_input']
                encoder_hidden_states = batch['encoder_hidden_states']
                timesteps = batch['timesteps']
                noise = batch['noise']
                model_pred = self.model.apply(
                        { 'params': params },
                        latent_model_input,
                        timesteps,
                        encoder_hidden_states
                ).sample
                loss = (noise - model_pred) ** 2
                loss = loss.mean()
                return loss
            grad_fn = jax.value_and_grad(compute_loss)

            def loss_and_grad(
                    train_rng: jax.random.PRNGKeyArray
            ) -> Tuple[jax.Array, Any, jax.random.PRNGKeyArray]:
                sample_rng, train_rng = jax.random.split(train_rng, 2)
                loss, grad = grad_fn(state.params, batch, sample_rng)
                return loss, grad, train_rng

            loss, grad, new_train_rng = loss_and_grad(train_rng)
            # self.log(grad) # NOTE uncomment to visualize gradient
            grad = jax.lax.pmean(grad, axis_name = 'batch')
            new_state = state.apply_gradients(grads = grad)
            metrics: Dict[str, Any] = { 'loss': loss }
            metrics = jax.lax.pmean(metrics, axis_name = 'batch')
            def l2(xs) -> jax.Array:
                return jnp.sqrt(sum([jnp.vdot(x, x) for x in jax.tree_util.tree_leaves(xs)]))
            metrics['l2_grads'] = l2(jax.tree_util.tree_leaves(grad))

            return new_state, metrics, new_train_rng

        p_train_step = jax.pmap(fun = train_step, axis_name = 'batch', donate_argnums = (0, ))
        state = jax_utils.replicate(state)

        train_metrics = []
        train_metric = None

        global_step: int = 0

        if jax.process_index() == 0:
            self._init_tracker_meta()
            hyper_params = {
                    'lr': lr,
                    'lr_warmup': warmup,
                    'lr_decay': decay,
                    'weight_decay': weight_decay,
                    'total_steps': total_steps,
                    'batch_size': dataloader.batch_size // self.num_devices,
                    'num_frames': num_frames,
                    'sample_size': self.sample_size,
                    'num_devices': self.num_devices,
                    'seed': self.seed,
                    'use_memory_efficient_attention': self.model.use_memory_efficient_attention,
                    'only_temporal': self.only_temporal,
                    'dtype': self.dtype_str,
                    'param_dtype': self.param_dtype,
                    'pretrained_model': self.pretrained_model,
                    'model_config': self.model.config
            }
            if self._use_wandb:
                self.log('Setting up wandb')
                self._setup_wandb(hyper_params)
            self.log(hyper_params)
            output_path = os.path.join(output_dir, str(global_step), 'unet')
            self.log(f'saving checkpoint to {output_path}')
            self.model.save_pretrained(
                    save_directory = output_path,
                    params = self.get_params_from_state(state),#jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state.params)),
                    is_main_process = True
            )

        pbar_epoch = tqdm(
                total = epochs,
                desc = 'Epochs',
                smoothing = 1,
                position = 0,
                dynamic_ncols = True,
                leave = True,
                disable = jax.process_index() > 0
        )
        steps_per_epoch = len(dataloader) # TODO dataloader
        for epoch in range(epochs):
            pbar_steps = tqdm(
                    total = steps_per_epoch,
                    desc = 'Steps',
                    position = 1,
                    smoothing = 0.1,
                    dynamic_ncols = True,
                    leave = True,
                    disable = jax.process_index() > 0
            )
            for batch in dataloader:
                # keep input + gt as float32, results in fp32 loss and grad
                # otherwise uncomment the following to cast to the model dtype
                # batch = { k: (v.astype(self.dtype) if v.dtype == np.float32 else v) for k,v in batch.items() }
                batch = shard(batch)
                state, train_metric, train_rngs = p_train_step(
                        state, batch, train_rngs
                )
                train_metrics.append(train_metric)
                if global_step % log_every_step == 0 and jax.process_index() == 0:
                    train_metrics = jax_utils.unreplicate(train_metrics)
                    train_metrics = jax.tree_util.tree_map(lambda *m: jnp.array(m).mean(), *train_metrics)
                    if global_step == 0:
                        self.log(f'grad dtype: {train_metrics["l2_grads"].dtype}')
                        self.log(f'loss dtype: {train_metrics["loss"].dtype}')
                    train_metrics_dict = { k: v.item() for k, v in train_metrics.items() }
                    train_metrics_dict['lr'] = lr_schedule(global_step).item()
                    self.log_metrics(train_metrics_dict, step = global_step, epoch = epoch)
                    train_metrics = []
                pbar_steps.update(1)
                global_step += 1
            if epoch % save_every_epoch == 0 and jax.process_index() == 0:
                output_path = os.path.join(output_dir, str(global_step), 'unet')
                self.log(f'saving checkpoint to {output_path}')
                self.model.save_pretrained(
                        save_directory = output_path,
                        params = self.get_params_from_state(state),#jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state.params)),
                        is_main_process = True
                )
                self.log(f'checkpoint saved ')
            if epoch % sample_every_epoch == 0 and jax.process_index() == 0:
                images = self.sample(
                        params = state.params,
                        replicate_params = False,
                        prompt = 'dancing person',
                        image_path = 'testimage.png',
                        num_frames = num_frames,
                        steps = 50,
                        cfg = 9.0,
                        unload_after_usage = False
                )
                os.makedirs(os.path.join('image_output', str(epoch)), exist_ok = True)
                for i, im in enumerate(images):
                    im.save(os.path.join('image_output', str(epoch), str(i).zfill(5) + '.png'), optimize = True)
            pbar_epoch.update(1)

