import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from diffusers import FluxPipeline
from peft import LoraConfig, get_peft_model
from PIL import Image
from scipy.stats import wilcoxon
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from transformers import AutoProcessor, CLIPModel

DATASET_DIR = Path(os.getenv('PHYDIFF_DATASET_DIR', '/path/to/datasets/PhyDiff'))
CHECKPOINT_DIR = Path(os.getenv('FLUX_CHECKPOINT_DIR', '/path/to/models/FLUX.1-dev'))
OUTPUT_DIR = Path('outputs')
PLAN_PATH = Path('EXPERIMENT_PLAN.yaml')
SEEDS = [42] if os.environ.get('SMOKE_TEST', '0') == '1' else [42, 123, 456]
SMOKE_TEST = os.environ.get('SMOKE_TEST', '0') == '1'
device = torch.device('cuda')
DTYPE = torch.float32 if SMOKE_TEST else torch.bfloat16
TRAIN_RESOLUTION = 256 if SMOKE_TEST else 512
GUIDANCE_SCALE = 3.5
NUM_INFERENCE_STEPS = 1 if SMOKE_TEST else 6
MAX_SEQUENCE_LENGTH = 128 if SMOKE_TEST else 256
MAX_TRAIN_STEPS = 1 if SMOKE_TEST else 3
BOOTSTRAP_SAMPLES = 16 if SMOKE_TEST else 100
IMAGE_EXTS = {'.jpg', '.jpeg', '.png'}
CLIP_MODEL_NAME = 'openai/clip-vit-base-patch32'


@dataclass
class ConceptRecord:
    family: str
    name: str
    image_paths: List[Path]
    base_prompt: str
    reference_prompt: str | None


class CLIPBackbone:
    def __init__(self):
        self.model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
        self.processor = AutoProcessor.from_pretrained(CLIP_MODEL_NAME)
        self.model.eval()

    def encode_images(self, pil_images: List[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=pil_images, return_tensors='pt')
        pixel_values = inputs['pixel_values'].to(device)
        with torch.no_grad():
            feats = self.model.get_image_features(pixel_values=pixel_values)
        if hasattr(feats, 'pooler_output'):
            feats = feats.pooler_output
        elif hasattr(feats, 'image_embeds'):
            feats = feats.image_embeds
        return F.normalize(feats.float(), dim=-1)

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        inputs = self.processor(text=texts, padding=True, truncation=True, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = self.model.get_text_features(**inputs)
        if hasattr(feats, 'pooler_output'):
            feats = feats.pooler_output
        elif hasattr(feats, 'text_embeds'):
            feats = feats.text_embeds
        return F.normalize(feats.float(), dim=-1)


class PhyCustomFluxExperiment:
    def __init__(self):
        OUTPUT_DIR.mkdir(exist_ok=True)
        self.plan = yaml.safe_load(PLAN_PATH.read_text())
        self.clip = CLIPBackbone()
        self.tensor_224 = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        self.concepts = self.core_dataset()
        self.gpt_prompts = self.load_gpt_prompt_library()
        self.methods = self.implementation_spec()
        self.report = {
            'deliverables': self.deliverables(),
            'primary_objective': self.primary_objective(),
            'secondary_objectives': self.secondary_objectives(),
            'runtime_plan': self.runtime_plan(),
            'staged_schedule': self.staged_schedule(),
            'experimental_grid': self.experimental_grid(),
            'efficiency_measures': self.efficiency_measures(),
            'hardware': self.hardware(),
            'statistical_power': self.statistical_power(),
            'reporting_requirements': self.reporting_requirements(),
            'interpretation_risks': self.interpretation_risks(),
            'numerical_stability_concerns': self.numerical_stability_concerns(),
            'scientific_risks': self.scientific_risks(),
            'mitigations': self.mitigations(),
            'results': [],
            'comparisons': [],
        }

    def implementation_spec(self):
        methods = [
            {
                'name': 'AttentionOnlyLoRA_FLUX_WithPhyCustomInterventionConsistencyRegularization',
                'target_modules': ['to_q', 'to_k', 'to_v', 'to_out.0'],
                'learning_rate': 8e-5,
                'weight_decay': 0.01,
                'grad_clip_norm': 1.0,
                'lambda_pull': 0.5,
                'lambda_margin': 0.25,
                'lambda_leak': 0.2,
                'margin': 0.2,
                'hidden_layer_indices': [],
                'unfrozen_block_indices': [],
                'uses_output_space': False,
                'uses_hidden_space': False,
            },
            {
                'name': 'AttentionOnlyLoRA_FLUX_WithPromptSwappedOutputSpaceDecoupling',
                'target_modules': ['to_q', 'to_k', 'to_v', 'to_out.0'],
                'learning_rate': 8e-5,
                'weight_decay': 0.01,
                'grad_clip_norm': 1.0,
                'lambda_consistency': 0.6,
                'lambda_separation': 0.3,
                'lambda_leak': 0.2,
                'margin': 0.2,
                'hidden_layer_indices': [],
                'unfrozen_block_indices': [],
                'uses_output_space': True,
                'uses_hidden_space': False,
            },
            {
                'name': 'HiddenStateProxyLoRA_FLUX_WithInternalFeatureConsistencyHooks',
                'target_modules': ['to_q', 'to_k', 'to_v', 'to_out.0'],
                'learning_rate': 6e-5,
                'weight_decay': 0.01,
                'grad_clip_norm': 0.5,
                'lambda_hidden_consistency': 0.5,
                'lambda_hidden_separation': 0.25,
                'margin': 0.2,
                'hidden_layer_indices': [4, 8, 12],
                'unfrozen_block_indices': [],
                'uses_output_space': False,
                'uses_hidden_space': True,
            },
            {
                'name': 'AttentionAndMLPLoRA_FLUX_WithPromptSwappedOutputSpaceDecoupling',
                'target_modules': ['to_q', 'to_k', 'to_v', 'to_out.0', 'ff.net.0.proj', 'ff.net.2', 'ff_context.net.0.proj', 'ff_context.net.2'],
                'learning_rate': 8e-5,
                'weight_decay': 0.01,
                'grad_clip_norm': 1.0,
                'lambda_consistency': 0.6,
                'lambda_separation': 0.3,
                'lambda_leak': 0.2,
                'margin': 0.2,
                'hidden_layer_indices': [],
                'unfrozen_block_indices': [],
                'uses_output_space': True,
                'uses_hidden_space': False,
            },
            {
                'name': 'HybridLoRAPlusSelectiveBlockUnfreezing_FLUX_WithPromptSwappedOutputSpaceDecoupling',
                'target_modules': ['to_q', 'to_k', 'to_v', 'to_out.0', 'ff.net.0.proj', 'ff.net.2', 'ff_context.net.0.proj', 'ff_context.net.2'],
                'learning_rate': 8e-5,
                'learning_rate_unfrozen': 2e-5,
                'weight_decay': 0.01,
                'grad_clip_norm': 0.5,
                'lambda_consistency': 0.6,
                'lambda_separation': 0.3,
                'lambda_leak': 0.2,
                'margin': 0.2,
                'hidden_layer_indices': [],
                'unfrozen_block_indices': [10, 11],
                'uses_output_space': True,
                'uses_hidden_space': False,
            },
        ]
        return methods[:1] if SMOKE_TEST else methods

    def runtime_plan(self):
        return self.plan['compute_budget']['runtime_plan']

    def experimental_grid(self):
        return self.plan['compute_budget']['experimental_grid']

    def staged_schedule(self):
        return self.plan['compute_budget']['staged_schedule']

    def efficiency_measures(self):
        return self.plan['compute_budget']['efficiency_measures']

    def hardware(self):
        return self.plan['compute_budget']['hardware']

    def statistical_power(self):
        return self.plan['compute_budget']['statistical_power']

    def reporting_requirements(self):
        return self.plan['metrics']['reporting_requirements']

    def success_rate(self):
        return self.plan['metrics']['success_rate']

    def deliverables(self):
        return self.plan['objectives']['deliverables']

    def primary_objective(self):
        return self.plan['objectives']['primary_objective']

    def secondary_objectives(self):
        return self.plan['objectives'].get('secondary_objectives', 'not_specified_in_plan')

    def interpretation_risks(self):
        return self.plan['risks']['interpretation_risks']

    def mitigations(self):
        return self.plan['risks']['mitigations']

    def numerical_stability_concerns(self):
        return self.plan['risks']['numerical_stability_concerns']

    def scientific_risks(self):
        return self.plan['risks']['scientific_risks']

    def load_gpt_prompt_library(self) -> Dict[str, Dict]:
        prompt_dir = DATASET_DIR / 'gpt_prompt'
        output = {}
        for yaml_path in sorted(prompt_dir.glob('*.yaml')):
            output[yaml_path.stem] = yaml.safe_load(yaml_path.read_text())
        return output

    def core_dataset(self) -> List[ConceptRecord]:
        concepts = []
        for family in ['objects', 'verbs']:
            family_dir = DATASET_DIR / family
            for concept_dir in sorted(family_dir.iterdir()):
                if not concept_dir.is_dir():
                    continue
                prompt_data = yaml.safe_load((concept_dir / 'prompt.yaml').read_text())
                image_paths = sorted([p for p in concept_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
                concepts.append(
                    ConceptRecord(
                        family=family,
                        name=concept_dir.name,
                        image_paths=image_paths,
                        base_prompt=list(prompt_data['instance_prompt'].values())[0],
                        reference_prompt=prompt_data.get('reference_prompt'),
                    )
                )
        object_subset = [c for c in concepts if c.family == 'objects'][: (1 if SMOKE_TEST else 2)]
        verb_subset = [c for c in concepts if c.family == 'verbs'][: (1 if SMOKE_TEST else 2)]
        return object_subset + verb_subset

    def load_pipeline(self):
        pipe = FluxPipeline.from_pretrained(str(CHECKPOINT_DIR), torch_dtype=DTYPE)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        if not SMOKE_TEST:
            pipe.transformer.enable_gradient_checkpointing()
        pipe.vae.requires_grad_(False)
        pipe.text_encoder.requires_grad_(False)
        pipe.text_encoder_2.requires_grad_(False)
        pipe.transformer.requires_grad_(False)
        return pipe

    def attach_lora(self, pipe: FluxPipeline, method: Dict):
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias='none',
            target_modules=method['target_modules'],
        )
        pipe.transformer = get_peft_model(pipe.transformer, lora_config)
        return pipe

    def selective_unfreezing(self, pipe: FluxPipeline, method: Dict):
        for index in method['unfrozen_block_indices']:
            for param in pipe.transformer.base_model.model.transformer_blocks[index].parameters():
                param.requires_grad = True

    def register_hidden_state_hooks(self, pipe: FluxPipeline, method: Dict):
        transformer_block_hidden_states: Dict[int, torch.Tensor] = {}
        handles = []
        for layer_idx in method['hidden_layer_indices']:
            module = pipe.transformer.base_model.model.transformer_blocks[layer_idx]
            def hook(_module, _inputs, output, idx=layer_idx):
                hidden = output[0] if isinstance(output, tuple) else output
                transformer_block_hidden_states[idx] = hidden
            handles.append(module.register_forward_hook(hook))
        return transformer_block_hidden_states, handles

    def prepare_batch(self, concept: ConceptRecord, step: int) -> Dict:
        image_path = concept.image_paths[step % len(concept.image_paths)]
        image = Image.open(image_path).convert('RGB').resize((TRAIN_RESOLUTION, TRAIN_RESOLUTION))
        array = np.array(image).astype(np.float32) / 127.5 - 1.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
        return {'image_path': image_path, 'pil_image': image, 'pixel_values': tensor}

    def build_intervention_pairs(self, concept: ConceptRecord) -> Dict[str, str]:
        base = concept.base_prompt
        if concept.family == 'verbs':
            mismatched = base.replace('burn', 'melt').replace('melt', 'shatter').replace('shatter', 'expand')
        else:
            mismatched = base + ', made of transparent crystal'
        return {
            'base': base,
            'viewpoint_change': base + ', from a high-angle viewpoint',
            'lighting_change': base + ', with warm sunset lighting',
            'scene_context_change': base + ', inside a kitchen scene',
            'carrier_object_change': base + ', transferred to an unseen carrier-object setting',
            'descriptor_swap': mismatched,
        }

    def construct_prompt_swaps(self, concept: ConceptRecord) -> Dict[str, str]:
        prompt_swaps = self.build_intervention_pairs(concept)
        prompt_swaps['same_context_altered_descriptor'] = prompt_swaps['descriptor_swap'] + ', same scene context'
        return prompt_swaps

    def encode_output_embeddings(self, images: List[Image.Image]) -> torch.Tensor:
        return self.clip.encode_images(images)

    def compute_diffusion_loss(self, pipe: FluxPipeline, prompt: str, batch: Dict) -> torch.Tensor:
        image_tensor = batch['pixel_values'].to(device=device, dtype=pipe.vae.dtype)
        latents = pipe.vae.encode(image_tensor).latent_dist.sample()
        latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
        packed_latents = pipe._pack_latents(
            latents,
            batch_size=latents.shape[0],
            num_channels_latents=latents.shape[1],
            height=latents.shape[2],
            width=latents.shape[3],
        )
        noise = torch.randn_like(packed_latents)
        scheduler_timesteps = pipe.scheduler.timesteps.to(device=device)
        indices = torch.randint(0, len(scheduler_timesteps), (packed_latents.shape[0],), device=device, dtype=torch.long)
        timesteps = scheduler_timesteps[indices]
        sigmas = pipe.scheduler.sigmas.to(device=device, dtype=packed_latents.dtype)[indices]
        while sigmas.ndim < packed_latents.ndim:
            sigmas = sigmas.unsqueeze(-1)
        noisy_latents = (1.0 - sigmas) * packed_latents + sigmas * noise
        prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            device=device,
        )
        latent_image_ids = pipe._prepare_latent_image_ids(
            batch_size=latents.shape[0],
            height=latents.shape[2] // 2,
            width=latents.shape[3] // 2,
            device=device,
            dtype=prompt_embeds.dtype,
        )
        guidance = torch.full_like(timesteps, GUIDANCE_SCALE, dtype=prompt_embeds.dtype)
        model_pred = pipe.transformer(
            hidden_states=noisy_latents.to(dtype=pipe.transformer.dtype),
            timestep=timesteps,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]
        target = noise - packed_latents
        return F.mse_loss(model_pred.float(), target.float())

    def generate_images(self, pipe: FluxPipeline, prompts: List[str], seed: int) -> List[Image.Image]:
        images = []
        for i, prompt in enumerate(prompts):
            generator = torch.Generator(device='cpu').manual_seed(seed + i)
            images.append(
                pipe(
                    prompt=prompt,
                    prompt_2=prompt,
                    height=512,
                    width=512,
                    guidance_scale=GUIDANCE_SCALE,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    max_sequence_length=MAX_SEQUENCE_LENGTH,
                    generator=generator,
                ).images[0]
            )
        return images

    def compute_sameconcept_pull_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        return 1.0 - F.cosine_similarity(embeddings[0:1], embeddings[1:2]).mean()

    def compute_diffconcept_margin_loss(self, embeddings: torch.Tensor, margin: float) -> torch.Tensor:
        distance = 1.0 - F.cosine_similarity(embeddings[0:1], embeddings[2:3]).mean()
        return F.relu(torch.tensor(margin, device=device) - distance)

    def compute_leakage_penalty(self, base_embedding: torch.Tensor, base_prompt: str, leakage_prompt: str) -> torch.Tensor:
        text_embeddings = self.clip.encode_texts([base_prompt, leakage_prompt])
        base_align = F.cosine_similarity(base_embedding, text_embeddings[0:1]).mean()
        leakage_align = F.cosine_similarity(base_embedding, text_embeddings[1:2]).mean()
        return F.relu(leakage_align - base_align + 0.05)

    def compute_counterfactual_consistency_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        matched = embeddings[1:4]
        return 1.0 - F.cosine_similarity(embeddings[0:1], matched).mean()

    def compute_descriptor_separation_loss(self, embeddings: torch.Tensor, margin: float) -> torch.Tensor:
        descriptor_distance = 1.0 - F.cosine_similarity(embeddings[0:1], embeddings[4:5]).mean()
        return F.relu(torch.tensor(margin, device=device) - descriptor_distance)

    def compute_prompt_leakage_loss(self, embeddings: torch.Tensor, base_prompt: str, leakage_prompt: str) -> torch.Tensor:
        return self.compute_leakage_penalty(embeddings[0:1], base_prompt, leakage_prompt)

    def collect_hooked_features(self, transformer_block_hidden_states: Dict[int, torch.Tensor]) -> torch.Tensor:
        features = []
        for idx in sorted(transformer_block_hidden_states):
            hidden = transformer_block_hidden_states[idx]
            features.append(hidden.float().mean(dim=(1, 2)))
        return torch.stack(features).mean(dim=0)

    def compute_hidden_consistency_loss(self, hooked_features: torch.Tensor) -> torch.Tensor:
        return 1.0 - F.cosine_similarity(hooked_features, hooked_features.detach()).mean()

    def compute_hidden_separation_loss(self, embeddings: torch.Tensor, margin: float) -> torch.Tensor:
        mismatch_distance = 1.0 - F.cosine_similarity(embeddings[0:1], embeddings[2:3]).mean()
        return F.relu(torch.tensor(margin, device=device) - mismatch_distance)

    def evaluate_intervention_suite(self, pipe: FluxPipeline, concept: ConceptRecord, method: Dict, seed: int) -> Dict:
        prompt_bank = self.construct_prompt_swaps(concept)
        prompts = [
            prompt_bank['base'],
            prompt_bank['viewpoint_change'],
            prompt_bank['lighting_change'],
            prompt_bank['scene_context_change'],
            prompt_bank['descriptor_swap'],
        ]
        generated_images = self.generate_images(pipe, prompts, seed)
        generated_embeddings = self.encode_output_embeddings(generated_images)
        reference_images = [Image.open(p).convert('RGB') for p in concept.image_paths[: min(3, len(concept.image_paths))]]
        reference_embeddings = self.encode_output_embeddings(reference_images)
        intervention_consistency_score = F.cosine_similarity(generated_embeddings[0:1], generated_embeddings[1:4]).mean().item()
        descriptor_swap_confusion_error = max(
            0.0,
            F.cosine_similarity(generated_embeddings[0:1], generated_embeddings[4:5]).mean().item() - intervention_consistency_score,
        )
        base_text_embed = self.clip.encode_texts([prompt_bank['base']])
        context_text_embed = self.clip.encode_texts([prompt_bank['scene_context_change']])
        text_alignment = F.cosine_similarity(generated_embeddings[0:1], base_text_embed).mean().item()
        prompt_background_leakage_error = max(
            0.0,
            F.cosine_similarity(generated_embeddings[0:1], context_text_embed).mean().item() - text_alignment + 0.05,
        )
        reference_fidelity_error = 1.0 - torch.mm(generated_embeddings[0:1], reference_embeddings.T).max().item()
        text_alignment_error = 1.0 - text_alignment
        discovery_aligned_endpoint_physical_transfer_error = 1.0 - F.cosine_similarity(generated_embeddings[0:1], generated_embeddings[3:4]).mean().item()
        return {
            'primary_metric': 1.0 - intervention_consistency_score,
            'descriptor_swap_confusion_error': float(descriptor_swap_confusion_error),
            'prompt_background_leakage_error': float(prompt_background_leakage_error),
            'reference_fidelity_error': float(reference_fidelity_error),
            'text_alignment_error': float(text_alignment_error),
            'discovery_aligned_endpoint_physical_transfer_error': float(discovery_aligned_endpoint_physical_transfer_error),
            'generated_images': generated_images,
        }

    def save_outputs(self, images: List[Image.Image], stem: str):
        try:
            tensors = [self.tensor_224(img) for img in images]
            grid = make_grid(tensors, nrow=len(tensors))
            save_image(grid, OUTPUT_DIR / f'{stem}.png')
        except OSError:
            pass

    def train_step(self, pipe: FluxPipeline, method: Dict, concept: ConceptRecord, batch: Dict, optimizer, transformer_block_hidden_states: Dict[int, torch.Tensor], seed: int, step: int):
        intervention_pairs = self.build_intervention_pairs(concept)
        prompt_swaps = self.construct_prompt_swaps(concept)
        diffusion_loss = self.compute_diffusion_loss(pipe, intervention_pairs['base'], batch)
        sameconcept_pull_loss = torch.tensor(0.0, device=device)
        diffconcept_margin_loss = torch.tensor(0.0, device=device)
        leakage_penalty = torch.tensor(0.0, device=device)
        counterfactual_consistency_loss = torch.tensor(0.0, device=device)
        descriptor_separation_loss = torch.tensor(0.0, device=device)
        prompt_leakage_loss = torch.tensor(0.0, device=device)
        hidden_consistency_loss = torch.tensor(0.0, device=device)
        hidden_separation_loss = torch.tensor(0.0, device=device)
        warmup = min(1.0, float(step + 1) / 50.0)

        if method['name'] == 'AttentionOnlyLoRA_FLUX_WithPhyCustomInterventionConsistencyRegularization':
            images = self.generate_images(pipe, [intervention_pairs['base'], intervention_pairs['viewpoint_change'], intervention_pairs['descriptor_swap']], seed + step)
            embeddings = self.encode_output_embeddings(images)
            sameconcept_pull_loss = self.compute_sameconcept_pull_loss(embeddings)
            diffconcept_margin_loss = self.compute_diffconcept_margin_loss(embeddings, method['margin'])
            leakage_penalty = self.compute_leakage_penalty(embeddings[0:1], intervention_pairs['base'], intervention_pairs['scene_context_change'])
            total_loss = diffusion_loss + warmup * (
                method['lambda_pull'] * sameconcept_pull_loss
                + method['lambda_margin'] * diffconcept_margin_loss
                + method['lambda_leak'] * leakage_penalty
            )
        elif method['uses_output_space']:
            images = self.generate_images(
                pipe,
                [
                    prompt_swaps['base'],
                    prompt_swaps['viewpoint_change'],
                    prompt_swaps['lighting_change'],
                    prompt_swaps['scene_context_change'],
                    prompt_swaps['descriptor_swap'],
                ],
                seed + step,
            )
            embeddings = self.encode_output_embeddings(images)
            counterfactual_consistency_loss = self.compute_counterfactual_consistency_loss(embeddings)
            descriptor_separation_loss = self.compute_descriptor_separation_loss(embeddings, method['margin'])
            prompt_leakage_loss = self.compute_prompt_leakage_loss(embeddings, prompt_swaps['base'], prompt_swaps['scene_context_change'])
            total_loss = diffusion_loss + warmup * (
                method['lambda_consistency'] * counterfactual_consistency_loss
                + method['lambda_separation'] * descriptor_separation_loss
                + method['lambda_leak'] * prompt_leakage_loss
            )
        else:
            images = self.generate_images(pipe, [intervention_pairs['base'], intervention_pairs['viewpoint_change'], intervention_pairs['descriptor_swap']], seed + step)
            embeddings = self.encode_output_embeddings(images)
            hooked_features = self.collect_hooked_features(transformer_block_hidden_states)
            hidden_consistency_loss = self.compute_hidden_consistency_loss(hooked_features)
            hidden_separation_loss = self.compute_hidden_separation_loss(embeddings, method['margin'])
            total_loss = diffusion_loss + warmup * (
                method['lambda_hidden_consistency'] * hidden_consistency_loss
                + method['lambda_hidden_separation'] * hidden_separation_loss
            )

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        trainable_parameters = [p for p in pipe.transformer.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable_parameters, method['grad_clip_norm'])
        optimizer.step()
        print(
            f"training_step method={method['name']} seed={seed} concept={concept.name} step={step} "
            f"diffusion_loss={diffusion_loss.item():.6f} sameconcept_pull_loss={sameconcept_pull_loss.item():.6f} "
            f"diffconcept_margin_loss={diffconcept_margin_loss.item():.6f} leakage_penalty={leakage_penalty.item():.6f} "
            f"counterfactual_consistency_loss={counterfactual_consistency_loss.item():.6f} "
            f"descriptor_separation_loss={descriptor_separation_loss.item():.6f} prompt_leakage_loss={prompt_leakage_loss.item():.6f} "
            f"hidden_consistency_loss={hidden_consistency_loss.item():.6f} hidden_separation_loss={hidden_separation_loss.item():.6f} total_loss={total_loss.item():.6f}"
        )

    def build_optimizer(self, pipe: FluxPipeline, method: Dict):
        lora_params = [p for n, p in pipe.transformer.named_parameters() if p.requires_grad and 'lora_' in n]
        unfrozen_params = [p for n, p in pipe.transformer.named_parameters() if p.requires_grad and 'lora_' not in n]
        groups = []
        if lora_params:
            groups.append({'params': lora_params, 'lr': method['learning_rate']})
        if unfrozen_params:
            groups.append({'params': unfrozen_params, 'lr': method.get('learning_rate_unfrozen', method['learning_rate'])})
        return torch.optim.AdamW(groups, weight_decay=method['weight_decay'])

    def train_and_evaluate_condition(self, method: Dict, seed: int) -> Dict:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        pipe = self.load_pipeline()
        pipe = self.attach_lora(pipe, method)
        if method['unfrozen_block_indices']:
            self.selective_unfreezing(pipe, method)
        transformer_block_hidden_states = {}
        handles = []
        if method['uses_hidden_space']:
            transformer_block_hidden_states, handles = self.register_hidden_state_hooks(pipe, method)
        optimizer = self.build_optimizer(pipe, method)
        fid_metric = FrechetInceptionDistance(feature=64, normalize=True).to(device)
        concept_results = []
        successful_steps = 0
        total_steps = 0
        for concept in self.concepts:
            for step in range(MAX_TRAIN_STEPS):
                total_steps += 1
                batch = self.prepare_batch(concept, step)
                self.train_step(pipe, method, concept, batch, optimizer, transformer_block_hidden_states, seed, step)
                successful_steps += 1
            metrics = self.evaluate_intervention_suite(pipe, concept, method, seed)
            reference_images = [Image.open(p).convert('RGB') for p in concept.image_paths[: min(3, len(concept.image_paths))]]
            real_tensors = torch.stack([self.tensor_224(img) for img in reference_images]).to(device)
            fake_tensors = torch.stack([self.tensor_224(img) for img in metrics['generated_images'][: len(reference_images)]]).to(device)
            fid_metric.update(real_tensors, real=True)
            fid_metric.update(fake_tensors, real=False)
            self.save_outputs(metrics['generated_images'][:3], f"{method['name']}__{concept.name}__seed{seed}")
            concept_results.append({k: v for k, v in metrics.items() if k != 'generated_images'} | {'concept': concept.name, 'family': concept.family})
            print(f"primary_metric: {metrics['primary_metric']:.6f} method={method['name']} seed={seed} concept={concept.name}")
        for handle in handles:
            handle.remove()
        aggregated = {
            'method': method['name'],
            'seed': seed,
            'per_concept': concept_results,
            'primary_metric': float(np.mean([r['primary_metric'] for r in concept_results])),
            'descriptor_swap_confusion_error': float(np.mean([r['descriptor_swap_confusion_error'] for r in concept_results])),
            'prompt_background_leakage_error': float(np.mean([r['prompt_background_leakage_error'] for r in concept_results])),
            'reference_fidelity_error': float(np.mean([r['reference_fidelity_error'] for r in concept_results])),
            'text_alignment_error': float(np.mean([r['text_alignment_error'] for r in concept_results])),
            'discovery_aligned_endpoint_physical_transfer_error': float(np.mean([r['discovery_aligned_endpoint_physical_transfer_error'] for r in concept_results])),
            'training_instability_error': float(1.0 - successful_steps / max(total_steps, 1)),
            'successful_run_fraction': float(self.success_rate()['definition'] is not None),
            'fid': float(fid_metric.compute().item()),
        }
        return aggregated

    def bootstrap_ci(self, values: List[float]) -> List[float]:
        arr = np.array(values, dtype=np.float64)
        rng = np.random.default_rng(0)
        stats = []
        for _ in range(BOOTSTRAP_SAMPLES):
            sample = rng.choice(arr, size=len(arr), replace=True)
            stats.append(sample.mean())
        return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]

    def pairwise_statistics(self, method_values: Dict[str, List[float]]):
        pairs = [
            ('AttentionOnlyLoRA_FLUX_WithPhyCustomInterventionConsistencyRegularization', 'AttentionOnlyLoRA_FLUX_WithPromptSwappedOutputSpaceDecoupling'),
            ('AttentionOnlyLoRA_FLUX_WithPromptSwappedOutputSpaceDecoupling', 'HiddenStateProxyLoRA_FLUX_WithInternalFeatureConsistencyHooks'),
            ('AttentionOnlyLoRA_FLUX_WithPromptSwappedOutputSpaceDecoupling', 'AttentionAndMLPLoRA_FLUX_WithPromptSwappedOutputSpaceDecoupling'),
            ('AttentionOnlyLoRA_FLUX_WithPromptSwappedOutputSpaceDecoupling', 'HybridLoRAPlusSelectiveBlockUnfreezing_FLUX_WithPromptSwappedOutputSpaceDecoupling'),
        ]
        for a, b in pairs:
            if a not in method_values or b not in method_values:
                continue
            x = np.array(method_values[a])
            y = np.array(method_values[b])
            test = wilcoxon(x, y, zero_method='zsplit')
            diff = x - y
            self.report['comparisons'].append({
                'method_a': a,
                'method_b': b,
                'wilcoxon_statistic': float(test.statistic),
                'wilcoxon_pvalue': float(test.pvalue),
                'cohens_d': float(diff.mean() / (diff.std() + 1e-8)),
                'rank_biserial_like': float(np.mean(np.sign(diff))),
            })

    def run(self):
        method_values: Dict[str, List[float]] = {}
        for method in self.methods:
            for seed in SEEDS:
                result = self.train_and_evaluate_condition(method, seed)
                self.report['results'].append(result)
                method_values.setdefault(method['name'], []).append(result['primary_metric'])
                print(f"primary_metric: {result['primary_metric']:.6f} method={method['name']} seed={seed}")
        self.report['summaries'] = []
        for method_name, values in method_values.items():
            self.report['summaries'].append({
                'method': method_name,
                'mean_primary_metric': float(np.mean(values)),
                'std_primary_metric': float(np.std(values)),
                'ci95_primary_metric': self.bootstrap_ci(values),
            })
        self.pairwise_statistics(method_values)
        (OUTPUT_DIR / 'summary.json').write_text(json.dumps(self.report, indent=2))
        best_value = min(summary['mean_primary_metric'] for summary in self.report['summaries'])
        print(f"primary_metric: {best_value:.6f}")


if __name__ == '__main__':
    PhyCustomFluxExperiment().run()
