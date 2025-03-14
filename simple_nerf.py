# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Implementation of simple nerf.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Tuple, Type

import torch
from torch.nn import Parameter

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.field_components.encodings import FreeNeRFEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.temporal_distortions import TemporalDistortionKind
from nerfstudio.fields.vanilla_nerf_field import NeRFField
from nerfstudio.model_components.losses import MSELoss, scale_gradients_by_distance_squared, DepthLossType, depth_loss, depth_ranking_loss
from nerfstudio.model_components.losses import course_fine_consistensy_loss, points_course_loss, view_course_loss
from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, RGBRenderer
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, misc


@dataclass
class SimpleNeRFModelConfig(ModelConfig):
    """SimpleNeRF Model Config"""

    _target: Type = field(default_factory=lambda: SimpleNeRFModel)
    num_coarse_samples: int = 64
    """Number of samples in coarse field evaluation"""
    num_importance_samples: int = 128
    """Number of samples in fine field evaluation"""

    enable_temporal_distortion: bool = False
    """Specifies whether or not to include ray warping based on time."""
    temporal_distortion_params: Dict[str, Any] = to_immutable_dict({"kind": TemporalDistortionKind.DNERF})
    """Parameters to instantiate temporal distortion with"""
    use_gradient_scaling: bool = False
    """Use gradient scaler where the gradients are lower for points closer to the camera."""
    background_color: Literal["random", "last_sample", "black", "white"] = "white"
    """Whether to randomize the background color."""


class SimpleNeRFModel(Model):
    """SimpleNeRF NeRF model

    Args:
        config: SimpleNeRF configuration to instantiate model
    """

    config: SimpleNeRFModelConfig

    def __init__(
        self,
        config: SimpleNeRFModelConfig,
        **kwargs,
    ) -> None:
        self.field_coarse = None
        self.field_fine = None
        self.temporal_distortion = None

        super().__init__(
            config=config,
            **kwargs,
        )

    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()

        # fields
        position_encoding = FreeNeRFEncoding(
            in_dim=3, max_iters=100000,num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        )
        augmented_position_encoding = FreeNeRFEncoding(
            in_dim=3,max_iters=100000, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=4.0, include_input=True
        )
        direction_encoding = FreeNeRFEncoding(
            in_dim=3,max_iters=100000, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=4.0, include_input=True
        )

        view_invariant_direction_encoding = FreeNeRFEncoding(
            in_dim=3,max_iters=100000, num_frequencies=0, min_freq_exp=0.0, max_freq_exp=0.0, include_input=False
        )

        self.field_coarse = NeRFField(
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
        )

        self.field_fine = NeRFField(
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
        )

        self.field_pos = NeRFField(
            position_encoding=augmented_position_encoding,
            direction_encoding=direction_encoding,
        )

        self.field_view = NeR
        FField(
            position_encoding=position_encoding,
            direction_encoding=view_invariant_direction_encoding,
        )

        # samplers
        self.sampler_uniform = UniformSampler(num_samples=self.config.num_coarse_samples)
        self.sampler_pdf = PDFSampler(num_samples=self.config.num_importance_samples)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()
        self.renderer_median_depth = DepthRenderer(method="median")

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        from torchmetrics.functional import structural_similarity_index_measure
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

        if getattr(self.config, "enable_temporal_distortion", False):
            params = self.config.temporal_distortion_params
            kind = params.pop("kind")
            self.temporal_distortion = kind.to_temporal_distortion(params)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        if self.field_coarse is None or self.field_fine is None:
            raise ValueError("populate_fields() must be called before get_param_groups")
        param_groups["fields"] = list(self.field_coarse.parameters()) + list(self.field_fine.parameters()) + list(self.field_pos.parameters()) + list(self.field_view.parameters())
        if self.temporal_distortion is not None:
            param_groups["temporal_distortion"] = list(self.temporal_distortion.parameters())
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):
        if self.field_coarse is None or self.field_fine is None:
            raise ValueError("populate_fields() must be called before get_outputs")

        # uniform sampling
        ray_samples_uniform = self.sampler_uniform(ray_bundle)
        if self.temporal_distortion is not None:
            offsets = None
            if ray_samples_uniform.times is not None:
                offsets = self.temporal_distortion(
                    ray_samples_uniform.frustums.get_positions(), ray_samples_uniform.times
                )
            ray_samples_uniform.frustums.set_offsets(offsets)

        # coarse field:
        field_outputs_coarse = self.field_coarse.forward(ray_samples_uniform)
        if self.config.use_gradient_scaling:
            field_outputs_coarse = scale_gradients_by_distance_squared(field_outputs_coarse, ray_samples_uniform)
        weights_coarse = ray_samples_uniform.get_weights(field_outputs_coarse[FieldHeadNames.DENSITY])
        rgb_coarse = self.renderer_rgb(
            rgb=field_outputs_coarse[FieldHeadNames.RGB],
            weights=weights_coarse,
        )
        accumulation_coarse = self.renderer_accumulation(weights_coarse)
        depth_coarse = self.renderer_median_depth(weights_coarse, ray_samples_uniform)

        # positional_encoding field:
        field_outputs_pos = self.field_pos.forward(ray_samples_uniform)
        if self.config.use_gradient_scaling:
            field_outputs_coarse = scale_gradients_by_distance_squared(field_outputs_coarse, ray_samples_uniform)
        weights_pos = ray_samples_uniform.get_weights(field_outputs_pos[FieldHeadNames.DENSITY])
        rgb_pos = self.renderer_rgb(
            rgb=field_outputs_pos[FieldHeadNames.RGB],
            weights=weights_pos,
        )
        accumulation_pos = self.renderer_accumulation(weights_pos)
        depth_pos = self.renderer_median_depth(weights_pos, ray_samples_uniform)
        
        # view invariant field:
        field_outputs_view = self.field_view.forward(ray_samples_uniform)
        if self.config.use_gradient_scaling:
            field_outputs_view = scale_gradients_by_distance_squared(field_outputs_view, ray_samples_uniform)
        weights_view = ray_samples_uniform.get_weights(field_outputs_view[FieldHeadNames.DENSITY])
        rgb_view = self.renderer_rgb(
            rgb=field_outputs_view[FieldHeadNames.RGB],
            weights=weights_view,
        )
        accumulation_view = self.renderer_accumulation(weights_view)
        depth_view = self.renderer_median_depth(weights_view, ray_samples_uniform)


        # pdf sampling
        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights_coarse)
        if self.temporal_distortion is not None:
            offsets = None
            if ray_samples_pdf.times is not None:
                offsets = self.temporal_distortion(ray_samples_pdf.frustums.get_positions(), ray_samples_pdf.times)
            ray_samples_pdf.frustums.set_offsets(offsets)

        # fine field:
        field_outputs_fine = self.field_fine.forward(ray_samples_pdf)
        if self.config.use_gradient_scaling:
            field_outputs_fine = scale_gradients_by_distance_squared(field_outputs_fine, ray_samples_pdf)
        weights_fine = ray_samples_pdf.get_weights(field_outputs_fine[FieldHeadNames.DENSITY])
        rgb_fine = self.renderer_rgb(
            rgb=field_outputs_fine[FieldHeadNames.RGB],
            weights=weights_fine,
        )
        accumulation_fine = self.renderer_accumulation(weights_fine)
        depth_fine = self.renderer_median_depth(weights_fine, ray_samples_pdf)

        outputs = {
            "rgb_coarse": rgb_coarse,
            "rgb_fine": rgb_fine,
            "rgb_pos": rgb_pos,
            "rgb_view": rgb_view,
            "accumulation_coarse": accumulation_coarse,
            "accumulation_fine": accumulation_fine,
            "accumulation_pos": accumulation_pos,
            "accumulation_view": accumulation_view,
            "depth_coarse": depth_coarse,
            "depth_fine": depth_fine,
            "depth_pos": depth_pos,
            "depth_view": depth_view,
            "weights_coarse": weights_coarse,
            "weights_fine": weights_fine,
            "weights_pos": weights_pos,
            "weights_view": weights_view,
            "ray_samples_uniform": ray_samples_uniform,
            "ray_samples_pdf": ray_samples_pdf,
        }
        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        # Scaling metrics by coefficients to create the losses.
        device = outputs["rgb_coarse"].device
        image = batch["image"].to(device)
        coarse_pred, coarse_image = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb_coarse"],
            pred_accumulation=outputs["accumulation_coarse"],
            gt_image=image,
        )
        fine_pred, fine_image = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb_fine"],
            pred_accumulation=outputs["accumulation_fine"],
            gt_image=image,
        )

        pos_pred, pos_image = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb_pos"],
            pred_accumulation=outputs["accumulation_pos"],
            gt_image=image,
        )

        view_pred, view_image = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb_view"],
            pred_accumulation=outputs["accumulation_view"],
            gt_image=image,
        )

        rgb_loss_coarse = self.rgb_loss(coarse_image, coarse_pred)
        rgb_loss_fine = self.rgb_loss(fine_image, fine_pred)
        rgb_loss_pos = self.rgb_loss(pos_image, pos_pred)
        rgb_loss_view = self.rgb_loss(view_image, view_pred)

        loss_dict = {"rgb_loss_coarse": rgb_loss_coarse, "rgb_loss_fine": rgb_loss_fine, "rgb_loss_pos": rgb_loss_pos, "rgb_loss_view": rgb_loss_view}
        
        termination_depth = batch["depth_image"].to(self.device)
        sigma = 0.01
        for i in range(len(outputs["weights_coarse"].shape[0])):
            loss_dict["depth_loss_course"] += depth_loss(
                weights=outputs["weights_course"][i],
                ray_samples=outputs["ray_samples_uniform"][i],
                termination_depth=termination_depth,
                sigma=sigma,
                depth_loss_type=DepthLossType.DS_NERF,
            ) / len(outputs["weights_coarse"].shape[0])
        
        for i in range(len(outputs["weights_fine"].shape[0])):
            loss_dict["depth_loss_fine"] += depth_loss(
                weights=outputs["weights_fine"][i],
                ray_samples=outputs["ray_samples_pdf"][i],
                termination_depth=termination_depth,
                sigma=sigma,
                depth_loss_type=DepthLossType.DS_NERF,
            ) / len(outputs["weights_fine"].shape[0])
        
        for i in range(len(outputs["weights_pos"].shape[0])):
            loss_dict["depth_loss_pos"] += depth_loss(
                weights=outputs["weights_pos"][i],
                ray_samples=outputs["ray_samples_uniform"][i],
                termination_depth=termination_depth,
                sigma=sigma,
                depth_loss_type=DepthLossType.DS_NERF,
            ) / len(outputs["weights_pos"].shape[0])
        
        for i in range(len(outputs["weights_view"].shape[0])):
            loss_dict["depth_loss_view"] += depth_loss(
                weights=outputs["weights_view"][i],
                ray_samples=outputs["ray_samples_uniform"][i],
                termination_depth=termination_depth,
                sigma=sigma,
                depth_loss_type=DepthLossType.DS_NERF,
            ) / len(outputs["weights_view"].shape[0])



        loss_dict["depth_loss_course"] = loss_dict["depth_loss_course"] * 1e-3
        loss_dict["depth_loss_fine"] = loss_dict["depth_loss_fine"] * 1e-3
        loss_dict["depth_loss_pos"] = loss_dict["depth_loss_pos"] * 1e-3
        loss_dict["depth_loss_view"] = loss_dict["depth_loss_view"] * 1e-3

        loss_dict["course_fine_consistency_loss"] = course_fine_consistensy_loss(loss_dict["depth_loss_fine"],loss_dict["depth_loss_course"])
        loss_dict["points_course_loss"] = points_course_loss(loss_dict["depth_loss_pos"],loss_dict["depth_loss_course"])
        loss_dict["view_course_loss"] = view_course_loss(loss_dict["depth_loss_view"],loss_dict["depth_loss_course"])
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(outputs["rgb_coarse"].device)
        image = self.renderer_rgb.blend_background(image)
        rgb_coarse = outputs["rgb_coarse"]
        rgb_fine = outputs["rgb_fine"]
        rgb_pos = outputs["rgb_pos"]
        rgb_view = outputs["rgb_view"]

        acc_coarse = colormaps.apply_colormap(outputs["accumulation_coarse"])
        acc_fine = colormaps.apply_colormap(outputs["accumulation_fine"])
        acc_pos = colormaps.apply_colormap(outputs["accumulation_pos"])
        acc_view = colormaps.apply_colormap(outputs["accumulation_view"])

        assert self.config.collider_params is not None
        depth_coarse = colormaps.apply_depth_colormap(
            outputs["depth_coarse"],
            accumulation=outputs["accumulation_coarse"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"],
        )
        depth_fine = colormaps.apply_depth_colormap(
            outputs["depth_fine"],
            accumulation=outputs["accumulation_fine"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"],
        )

        depth_pos = colormaps.apply_depth_colormap(
            outputs["depth_pos"],
            accumulation=outputs["accumulation_pos"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"],
        )

        depth_view = colormaps.apply_depth_colormap(
            outputs["depth_view"],
            accumulation=outputs["accumulation_view"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"],
        )

        combined_rgb = torch.cat([image, rgb_coarse, rgb_fine, rgb_pos, rgb_view], dim=1)
        combined_acc = torch.cat([acc_coarse, acc_fine, acc_pos, acc_view], dim=1)
        combined_depth = torch.cat([depth_coarse, depth_fine, depth_pos, depth_view], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb_coarse = torch.moveaxis(rgb_coarse, -1, 0)[None, ...]
        rgb_fine = torch.moveaxis(rgb_fine, -1, 0)[None, ...]

        coarse_psnr = self.psnr(image, rgb_coarse)
        fine_psnr = self.psnr(image, rgb_fine)
        fine_ssim = self.ssim(image, rgb_fine)
        fine_lpips = self.lpips(image, rgb_fine)
        assert isinstance(fine_ssim, torch.Tensor)

        metrics_dict = {
            "psnr": float(fine_psnr.item()),
            "coarse_psnr": float(coarse_psnr),
            "fine_psnr": float(fine_psnr),
            "fine_ssim": float(fine_ssim),
            "fine_lpips": float(fine_lpips),
        }
        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}
        return metrics_dict, images_dict
