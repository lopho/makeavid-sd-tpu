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

from typing import Any, Mapping, Dict


def map_2d_to_pseudo3d(
        params2d: Mapping[str, Any],
        params3d: Mapping[str, Any],
        verbose: bool = True
) -> Dict[str, Any]:
    new_params: Dict[str, Any] = dict()
    for k in params3d:
        kl = k.split('.')
        if 'spatial_conv' in kl:
            k2d = kl.copy()
            k2d.remove('spatial_conv')
            k2d = '.'.join(k2d)
            p = params2d[k2d]
            if verbose:
                print(f'Spatial: {k} <- {k2d}')
        elif k not in params2d:
            p = params3d[k]
            if verbose:
                print(f'Missing in 2D: {k}')
        else:
            p = params2d[k]
        assert p.shape == params3d[k].shape, f'shape mismatch: {k}: {p.shape} != {params3d[k].shape}'
        new_params[k] = p
    return new_params


def map_pseudo3d_to_2d(
        params2d: Mapping[str, Any],
        params3d: Mapping[str, Any],
        verbose: bool = True
) -> Dict[str, Any]:
    new_params: Dict[str, Any] = dict()
    for k in params3d:
        kl = k.split('.')
        if 'spatial_conv' in kl:
            k2d = kl.copy()
            k2d.remove('spatial_conv')
            new_k = '.'.join(k2d)
            if verbose:
                print(f'Spatial: {new_k} <- {k}')
        elif k not in params2d:
            new_k = None
            if verbose:
                print(f'Missing in 2D: {k}')
        else:
            new_k = k
        if new_k is not None:
            assert params3d[k].shape == params2d[new_k].shape, f'shape mismatch: {new_k}: {params2d[new_k].shape} != {params3d[k].shape}'
            new_params[new_k] = params3d[k]
    for k in params2d:
        if k not in new_params:
            if verbose:
                print(f'Missing in 3D: {k}')
            new_params[k] = params2d[k]
    return new_params

