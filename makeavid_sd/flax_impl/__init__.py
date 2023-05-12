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

from .flax_unet_pseudo3d_condition import (
        UNetPseudo3DConditionModel as FlaxUNetPseudo3DConditionModel,
        UNetPseudo3DConditionOutput as FlaxUNetPseudo3DConditionOutput
)

from .flax_resnet_pseudo3d import (
        ConvPseudo3D as FlaxConvPseudo3D,
        ResnetBlockPseudo3D as FlaxResnetBlockPseudo3D,
        DownsamplePseudo3D as FlaxDownsamplePseudo3D,
        UpsamplePseudo3D as FlaxUpsamplePseudo3D
)

from .flax_unet_pseudo3d_blocks import (
        UNetMidBlockPseudo3DCrossAttn as FlaxUNetMidBlockPseudo3DCrossAttn,
        CrossAttnUpBlockPseudo3D as FlaxCrossAttnUpBlockPseudo3D,
        CrossAttnDownBlockPseudo3D as FlaxCrossAttnDownBlockPseudo3D,
        DownBlockPseudo3D as FlaxDownBlockPseudo3D,
        UpBlockPseudo3D as FlaxUpBlockPseudo3D,
)

from .flax_attention_pseudo3d import (
        TransformerPseudo3DModel as FlaxTransformerPseudo3DModel,
        BasicTransformerBlockPseudo3D as FlaxBasicTransformerBlockPseudo3D,
        FeedForward as FlaxFeedForward,
        GEGLU as FlaxGEGLU
)

from .flax_trainer import (
        TrainerUNetPseudo3D as FlaxTrainerUNetPseudo3D
)

