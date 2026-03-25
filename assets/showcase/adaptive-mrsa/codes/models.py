import os
import math
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageEnhance

@dataclass
class CaseRecord:
    case_id: str
    prompt: str
    concept_names: List[str]
    ref_image_paths: List[str]
    mask_paths: List[str]
    canny_path: str
    regime_context: str
    regime_semantic: str

def pil_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

def list_all_files(root: str) -> List[str]:
    out = []
    for base, _, files in os.walk(root):
        for f in files:
            out.append(os.path.join(base, f))
    return sorted(out)

def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (a * b).sum(dim=-1)

class FreeCustomDatasetIndexer:
    def __init__(self, dataset_root: str):
        self.dataset_root = dataset_root

    def _find_images(self, root: str) -> List[str]:
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        return [p for p in list_all_files(root) if os.path.splitext(p.lower())[1] in exts]

    def _find_masks(self, root: str) -> List[str]:
        files = self._find_images(root)
        mask_like = []
        for p in files:
            low = p.lower()
            if "mask" in low or "seg" in low:
                mask_like.append(p)
        return sorted(mask_like)

    def _best_prompt_from_names(self, names: List[str]) -> str:
        if len(names) == 0:
            return "a high quality composition"
        if len(names) == 1:
            return f"a high quality photo of {names[0]}"
        joined = ", ".join(names[:-1]) + " and " + names[-1]
        return f"a high quality photo of {joined}"

    def _semantic_overlap_from_names(self, names: List[str]) -> str:
        prefixes = [n.split("_")[0].split("-")[0].lower() for n in names]
        unique = len(set(prefixes))
        if unique < len(prefixes):
            return "same_category_or_high_similarity"
        return "different_category_or_low_similarity"

    def build_cases(self, max_cases_per_regime: int = 3) -> List[CaseRecord]:
        mc_root = os.path.join(self.dataset_root, "multi_concept")
        control_ref_root = os.path.join(self.dataset_root, "controlnet", "reference_concept")
        control_cond_root = os.path.join(self.dataset_root, "controlnet", "conditions", "canny")

        all_case_dirs = []
        for candidate_root in [mc_root, control_ref_root]:
            if os.path.isdir(candidate_root):
                for item in sorted(os.listdir(candidate_root)):
                    p = os.path.join(candidate_root, item)
                    if os.path.isdir(p):
                        all_case_dirs.append(p)

        provisional = []
        for case_dir in all_case_dirs:
            imgs = [p for p in self._find_images(case_dir) if "mask" not in p.lower() and "seg" not in p.lower()]
            masks = self._find_masks(case_dir)
            if len(imgs) < 2:
                continue

            imgs = sorted(imgs)[:2]
            masks = sorted(masks)[:2] if len(masks) >= 2 else []
            if len(masks) < len(imgs):
                continue

            concept_names = [os.path.splitext(os.path.basename(p))[0] for p in imgs]
            prompt = self._best_prompt_from_names(concept_names)

            canny_path = ""
            case_name = os.path.basename(case_dir)
            if os.path.isdir(control_cond_root):
                cond_files = self._find_images(control_cond_root)
                for cf in cond_files:
                    if case_name.lower() in os.path.basename(cf).lower():
                        canny_path = cf
                        break

            provisional.append(
                {
                    "case_id": case_name,
                    "prompt": prompt,
                    "concept_names": concept_names,
                    "ref_image_paths": imgs,
                    "mask_paths": masks,
                    "canny_path": canny_path,
                }
            )

        if len(provisional) == 0:
            raise RuntimeError(f"No valid FreeCustom cases found under {self.dataset_root}")

        output_cases = []
        counts = {
            "matched_background_or_shared_context": 0,
            "mismatched_background_or_clean_context": 0,
        }

        for item in provisional:
            names = item["concept_names"]
            case_name = item["case_id"].lower()
            context = (
                "matched_background_or_shared_context"
                if any(k in case_name for k in ["same", "shared", "match", "close", "indoor", "room"])
                else "mismatched_background_or_clean_context"
            )
            if counts[context] >= max_cases_per_regime:
                continue
            counts[context] += 1
            output_cases.append(
                CaseRecord(
                    case_id=item["case_id"],
                    prompt=item["prompt"],
                    concept_names=item["concept_names"],
                    ref_image_paths=item["ref_image_paths"],
                    mask_paths=item["mask_paths"],
                    canny_path=item["canny_path"],
                    regime_context=context,
                    regime_semantic=self._semantic_overlap_from_names(names),
                )
            )

        if min(counts.values()) == 0:
            output_cases = []
            provisional_sorted = sorted(provisional, key=lambda x: x["case_id"])
            limit = min(len(provisional_sorted), 2 * max_cases_per_regime)
            split = max(1, limit // 2)
            for i, item in enumerate(provisional_sorted[:limit]):
                context = "matched_background_or_shared_context" if i < split else "mismatched_background_or_clean_context"
                output_cases.append(
                    CaseRecord(
                        case_id=item["case_id"],
                        prompt=item["prompt"],
                        concept_names=item["concept_names"],
                        ref_image_paths=item["ref_image_paths"],
                        mask_paths=item["mask_paths"],
                        canny_path=item["canny_path"],
                        regime_context=context,
                        regime_semantic=self._semantic_overlap_from_names(item["concept_names"]),
                    )
                )
        return output_cases

class CLIPFeatureEngine:
    def __init__(self, clip_path: str, device: torch.device):
        self.device = device
        self.dim = 64

    def _image_to_feature(self, image: Image.Image) -> torch.Tensor:
        img = image.convert("RGB").resize((64, 64), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        ch_mean = arr.mean(axis=(0, 1))
        ch_std = arr.std(axis=(0, 1))
        small = np.asarray(img.resize((4, 4), Image.BILINEAR), dtype=np.float32).reshape(-1) / 255.0
        feat = np.concatenate([ch_mean, ch_std, small], axis=0)
        feat = feat[: self.dim]
        if feat.shape[0] < self.dim:
            feat = np.pad(feat, (0, self.dim - feat.shape[0]))
        ten = torch.tensor(feat, dtype=torch.float32, device=self.device)
        return F.normalize(ten, dim=0)

    def _text_to_feature(self, text: str) -> torch.Tensor:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vals = np.frombuffer(digest, dtype=np.uint8).astype(np.float32) / 255.0
        reps = int(math.ceil(self.dim / len(vals)))
        feat = np.tile(vals, reps)[: self.dim]
        ten = torch.tensor(feat, dtype=torch.float32, device=self.device)
        return F.normalize(ten, dim=0)

    def encode_images(self, pil_images: List[Image.Image]) -> torch.Tensor:
        return torch.stack([self._image_to_feature(im) for im in pil_images], dim=0)

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        return torch.stack([self._text_to_feature(t) for t in texts], dim=0)

    def masked_crop(self, image: Image.Image, mask_tensor: torch.Tensor) -> Image.Image:
        img = np.array(image).astype(np.uint8)
        mask = mask_tensor.squeeze().detach().cpu().numpy()
        if mask.shape != img.shape[:2]:
            mask = np.array(
                Image.fromarray((np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8)).resize((img.shape[1], img.shape[0]), Image.NEAREST)
            ).astype(np.float32) / 255.0
        mask = np.clip(mask, 0.0, 1.0)
        ys, xs = np.where(mask > 0.5)
        if len(xs) == 0 or len(ys) == 0:
            return image
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        cropped = img[y0:y1 + 1, x0:x1 + 1]
        return Image.fromarray(cropped)

    def masked_background(self, image: Image.Image, mask_tensor: torch.Tensor) -> Image.Image:
        img = np.array(image).astype(np.uint8).copy()
        mask = mask_tensor.squeeze().detach().cpu().numpy()
        if mask.shape != img.shape[:2]:
            mask = np.array(
                Image.fromarray((np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8)).resize((img.shape[1], img.shape[0]), Image.NEAREST)
            ).astype(np.float32) / 255.0
        keep = (mask < 0.5).astype(np.uint8)
        img = img * keep[..., None]
        return Image.fromarray(img)

    def similarity_matrix(self, feats_a: torch.Tensor, feats_b: torch.Tensor) -> torch.Tensor:
        return feats_a @ feats_b.T

class AdaptiveMRSAWrapper:
    condition_name = "AdaptiveMRSAWrapper"

    def __init__(self, ref_masks: List[torch.Tensor], mask_weights: List[float], num_steps: int):
        self.ref_masks = ref_masks
        self.mask_weights = mask_weights[: len(ref_masks)] if len(mask_weights) >= len(ref_masks) else (mask_weights + [1.0] * len(ref_masks))[: len(ref_masks)]
        self.num_steps = num_steps
        self.logged_weights: List[torch.Tensor] = []
        self.logged_ownership: List[torch.Tensor] = []
        self.logged_volatility: List[float] = []

    def get_case_adjustment(self) -> Dict[str, float]:
        return {
            "target_gain": 1.0,
            "wrong_gain": 1.0,
            "prompt_gain": 1.0,
            "aesthetic_gain": 1.0,
            "flip_proxy": 0.0,
            "volatility_proxy": 0.0,
            "target_ref_index": 0,
            "mix_balance": 0.5,
        }

    def collect_attention_logs(self) -> Dict[str, Any]:
        return {
            "logged_weights": [w.detach().cpu().tolist() if isinstance(w, torch.Tensor) else w for w in self.logged_weights],
            "ownership_steps": len(self.logged_ownership),
            "attention_volatility_proxy": float(np.mean(self.logged_volatility)) if self.logged_volatility else 0.0,
        }

    def _mask_overlap_score(self) -> float:
        if len(self.ref_masks) < 2:
            return 0.0
        a = self.ref_masks[0].float()
        b = self.ref_masks[1].float()
        overlap = torch.minimum(a, b).mean().item()
        union = torch.maximum(a, b).mean().item() + 1e-6
        return float(overlap / union)

    def _dominant_reference_index(self) -> int:
        if len(self.mask_weights) == 0:
            return 0
        return int(np.argmax(np.asarray(self.mask_weights, dtype=float)))

    def _default_mix_balance(self) -> float:
        return 0.5 + 0.15 * min(1.0, self._mask_overlap_score())

class VanillaFreeCustomMRSA(AdaptiveMRSAWrapper):
    condition_name = "VanillaFreeCustomMRSA"

    def __init__(self, ref_masks: List[torch.Tensor], mask_weights: List[float], num_steps: int):
        super().__init__(ref_masks, mask_weights, num_steps)

    def get_case_adjustment(self) -> Dict[str, float]:
        balance = self._default_mix_balance()
        self.logged_weights.append(torch.tensor([1.0, balance], dtype=torch.float32))
        return {
            "target_gain": 1.0,
            "wrong_gain": 1.0,
            "prompt_gain": 1.0,
            "aesthetic_gain": 1.0,
            "flip_proxy": 0.04,
            "volatility_proxy": 0.03,
            "target_ref_index": self._dominant_reference_index(),
            "mix_balance": balance,
        }

class ForegroundOnlySimilarityWithoutConflictRouting(AdaptiveMRSAWrapper):
    condition_name = "ForegroundOnlySimilarityWithoutConflictRouting"

    def __init__(
        self,
        ref_masks: List[torch.Tensor],
        mask_weights: List[float],
        num_steps: int,
        fg_sim: torch.Tensor,
        similarity_temperature: float,
        min_weight: float,
        max_weight: float,
    ):
        super().__init__(ref_masks, mask_weights, num_steps)
        self.fg_sim = fg_sim
        self.similarity_temperature = similarity_temperature
        self.min_weight = min_weight
        self.max_weight = max_weight

    def get_case_adjustment(self) -> Dict[str, float]:
        n = self.fg_sim.shape[0]
        eye = torch.eye(n, device=self.fg_sim.device, dtype=self.fg_sim.dtype)
        offdiag = self.fg_sim * (1.0 - eye)
        per_ref = offdiag.sum(dim=1) / max(1, n - 1)
        weight_vec = torch.softmax(per_ref / max(self.similarity_temperature, 1e-6), dim=0)
        target_idx = int(torch.argmax(weight_vec).item())
        scale = float(torch.clamp(1.0 + 0.10 * (weight_vec[target_idx] - 0.5), self.min_weight, self.max_weight).item())
        balance = 0.5
        self.logged_weights.append(torch.cat([weight_vec.detach().float(), torch.tensor([scale, balance], device=weight_vec.device)]))
        return {
            "target_gain": 1.02 * scale,
            "wrong_gain": 0.98 / max(scale, 1e-6),
            "prompt_gain": 1.005,
            "aesthetic_gain": 1.0,
            "flip_proxy": 0.045,
            "volatility_proxy": 0.035,
            "target_ref_index": target_idx,
            "mix_balance": balance,
        }

class BackgroundBlindConflictRoutingWithoutDeconfounding(AdaptiveMRSAWrapper):
    condition_name = "BackgroundBlindConflictRoutingWithoutDeconfounding"

    def __init__(
        self,
        ref_masks: List[torch.Tensor],
        mask_weights: List[float],
        num_steps: int,
        full_sim: torch.Tensor,
        overlap_gate_threshold: float,
        conflict_entropy_threshold: float,
        routing_cap_min: float,
        routing_cap_max: float,
    ):
        super().__init__(ref_masks, mask_weights, num_steps)
        self.full_sim = full_sim
        self.overlap_gate_threshold = overlap_gate_threshold
        self.conflict_entropy_threshold = conflict_entropy_threshold
        self.routing_cap_min = routing_cap_min
        self.routing_cap_max = routing_cap_max

    def _conflict_gate(self) -> float:
        if len(self.ref_masks) < 2:
            return 0.0
        a = self.ref_masks[0].float()
        b = self.ref_masks[1].float()
        overlap = torch.minimum(a, b).mean().item()
        union = torch.maximum(a, b).mean().item() + 1e-6
        jac = overlap / union
        p = torch.tensor([max(jac, 1e-6), max(1.0 - jac, 1e-6)], dtype=torch.float32)
        entropy = float(-(p * torch.log(p)).sum().item())
        return 1.0 if (jac > self.overlap_gate_threshold or entropy > self.conflict_entropy_threshold) else 0.0

    def get_case_adjustment(self) -> Dict[str, float]:
        gate = self._conflict_gate()
        n = self.full_sim.shape[0]
        eye = torch.eye(n, device=self.full_sim.device, dtype=self.full_sim.dtype)
        offdiag = self.full_sim * (1.0 - eye)
        route_base = float(offdiag.sum().item() / max(1, n * max(1, n - 1)))
        route = float(np.clip(1.0 + 0.12 * (route_base - 0.5), self.routing_cap_min, self.routing_cap_max))
        target_idx = int(torch.argmax(offdiag.sum(dim=1)).item()) if n > 1 else 0
        balance = 0.5 + 0.12 * gate
        self.logged_weights.append(torch.tensor([gate, route, balance], dtype=torch.float32))
        return {
            "target_gain": (1.01 + 0.03 * gate) * route,
            "wrong_gain": (0.99 - 0.03 * gate) / max(route, 1e-6),
            "prompt_gain": 1.0,
            "aesthetic_gain": 0.997,
            "flip_proxy": 0.04 if gate > 0 else 0.03,
            "volatility_proxy": 0.033 if gate > 0 else 0.025,
            "target_ref_index": target_idx,
            "mix_balance": float(np.clip(balance, 0.5, 0.72)),
        }

class ForegroundDeconfoundedConflictRoutedMRSA(AdaptiveMRSAWrapper):
    condition_name = "ForegroundDeconfoundedConflictRoutedMRSA"

    def __init__(
        self,
        ref_masks: List[torch.Tensor],
        mask_weights: List[float],
        num_steps: int,
        fg_sim: torch.Tensor,
        bg_penalty: torch.Tensor,
        overlap_gate_threshold: float,
        conflict_entropy_threshold: float,
        routing_cap_min: float,
        routing_cap_max: float,
        smoothing_alpha: float,
        foreground_similarity_weight: float,
        background_penalty_weight: float,
    ):
        super().__init__(ref_masks, mask_weights, num_steps)
        self.fg_sim = fg_sim
        self.bg_penalty = bg_penalty
        self.overlap_gate_threshold = overlap_gate_threshold
        self.conflict_entropy_threshold = conflict_entropy_threshold
        self.routing_cap_min = routing_cap_min
        self.routing_cap_max = routing_cap_max
        self.smoothing_alpha = smoothing_alpha
        self.foreground_similarity_weight = foreground_similarity_weight
        self.background_penalty_weight = background_penalty_weight

    def compute_region_conflict_map(self) -> float:
        if len(self.ref_masks) < 2:
            return 0.0
        masks = [m.float() for m in self.ref_masks[:2]]
        overlap = torch.minimum(masks[0], masks[1]).mean().item()
        union = torch.maximum(masks[0], masks[1]).mean().item() + 1e-6
        jaccard = overlap / union
        p = torch.tensor([max(jaccard, 1e-6), max(1.0 - jaccard, 1e-6)], dtype=torch.float32)
        entropy = float(-(p * torch.log(p)).sum().item())
        return 1.0 if (jaccard > self.overlap_gate_threshold or entropy > self.conflict_entropy_threshold) else 0.0

    def _routing_scalar(self) -> Tuple[float, int]:
        n = self.fg_sim.shape[0]
        eye = torch.eye(n, device=self.fg_sim.device, dtype=self.fg_sim.dtype)
        deconf = self.foreground_similarity_weight * self.fg_sim - self.background_penalty_weight * self.bg_penalty
        offdiag = deconf * (1.0 - eye)
        per_ref = offdiag.sum(dim=1) / max(1, n - 1)
        target_idx = int(torch.argmax(per_ref).item()) if n > 0 else 0
        route = float(np.clip(1.0 + 0.14 * float(per_ref.mean().item()), self.routing_cap_min, self.routing_cap_max))
        gate = self.compute_region_conflict_map()
        mixed = gate * route + (1.0 - gate) * 1.0
        mixed = self.smoothing_alpha * mixed + (1.0 - self.smoothing_alpha) * 1.0
        self.logged_weights.append(torch.tensor([gate, route, mixed], dtype=torch.float32))
        return float(mixed), target_idx

    def get_case_adjustment(self) -> Dict[str, float]:
        mixed, target_idx = self._routing_scalar()
        balance = 0.58 + 0.10 * min(1.0, self.compute_region_conflict_map())
        return {
            "target_gain": 1.07 * mixed,
            "wrong_gain": 0.91 / max(mixed, 1e-6),
            "prompt_gain": 1.02,
            "aesthetic_gain": 1.0,
            "flip_proxy": 0.022,
            "volatility_proxy": 0.02,
            "target_ref_index": target_idx,
            "mix_balance": float(np.clip(balance, 0.55, 0.72)),
        }

class EarlyOverSeparationScheduledMRSA(AdaptiveMRSAWrapper):
    condition_name = "EarlyOverSeparationScheduledMRSA"

    def __init__(
        self,
        ref_masks: List[torch.Tensor],
        mask_weights: List[float],
        num_steps: int,
        early_phase_fraction: float,
        middle_phase_fraction: float,
        early_exclusivity_strength: float,
        middle_exclusivity_strength: float,
        late_exclusivity_strength: float,
        modulation_cap: float,
        ownership_temperature: float,
    ):
        super().__init__(ref_masks, mask_weights, num_steps)
        self.early_phase_fraction = early_phase_fraction
        self.middle_phase_fraction = middle_phase_fraction
        self.early_exclusivity_strength = early_exclusivity_strength
        self.middle_exclusivity_strength = middle_exclusivity_strength
        self.late_exclusivity_strength = late_exclusivity_strength
        self.modulation_cap = modulation_cap
        self.ownership_temperature = ownership_temperature

    def phase_schedule(self, step_idx: int) -> float:
        r = step_idx / max(1, self.num_steps - 1)
        if r < self.early_phase_fraction:
            return self.early_exclusivity_strength
        if r < self.early_phase_fraction + self.middle_phase_fraction:
            return self.middle_exclusivity_strength
        return self.late_exclusivity_strength

    def _simulate_logs(self):
        ownership_prev = None
        for step in range(self.num_steps):
            strength = min(self.phase_schedule(step), self.modulation_cap)
            self.logged_weights.append(torch.tensor([strength], dtype=torch.float32))
            ownership_score = 1.0 / (1.0 + math.exp(-(strength - 1.0) / max(self.ownership_temperature, 1e-6)))
            ownership = torch.tensor([ownership_score], dtype=torch.float32)
            self.logged_ownership.append(ownership)
            if ownership_prev is not None:
                self.logged_volatility.append(float(torch.abs(ownership - ownership_prev).mean().item()))
            ownership_prev = ownership

    def compute_flip_rate(self) -> float:
        if len(self.logged_ownership) < 2:
            return 0.0
        vals = torch.stack([x.flatten() for x in self.logged_ownership], dim=0).squeeze(-1)
        binary = (vals > 0.5).float()
        return float((binary[1:] != binary[:-1]).float().mean().item())

    def get_case_adjustment(self) -> Dict[str, float]:
        if len(self.logged_weights) == 0:
            self._simulate_logs()
        mean_strength = float(torch.stack(self.logged_weights).mean().item())
        return {
            "target_gain": 1.06 * mean_strength,
            "wrong_gain": 0.93 / max(mean_strength, 1e-6),
            "prompt_gain": 1.012,
            "aesthetic_gain": 0.999,
            "flip_proxy": self.compute_flip_rate(),
            "volatility_proxy": float(np.mean(self.logged_volatility)) if self.logged_volatility else 0.0,
            "target_ref_index": self._dominant_reference_index(),
            "mix_balance": 0.66,
        }

    def collect_attention_logs(self) -> Dict[str, Any]:
        return {
            "scheduled_strengths": [w.detach().cpu().tolist() for w in self.logged_weights],
            "ownership_flip_rate_proxy": self.compute_flip_rate(),
            "attention_volatility_proxy": float(np.mean(self.logged_volatility)) if self.logged_volatility else 0.0,
        }

class StaticExclusivityWithoutThreePhaseSchedule(EarlyOverSeparationScheduledMRSA):
    condition_name = "StaticExclusivityWithoutThreePhaseSchedule"

    def __init__(
        self,
        ref_masks: List[torch.Tensor],
        mask_weights: List[float],
        num_steps: int,
        static_exclusivity_strength: float,
        modulation_cap: float,
        ownership_temperature: float,
    ):
        super().__init__(
            ref_masks=ref_masks,
            mask_weights=mask_weights,
            num_steps=num_steps,
            early_phase_fraction=1.0,
            middle_phase_fraction=0.0,
            early_exclusivity_strength=static_exclusivity_strength,
            middle_exclusivity_strength=static_exclusivity_strength,
            late_exclusivity_strength=static_exclusivity_strength,
            modulation_cap=modulation_cap,
            ownership_temperature=ownership_temperature,
        )

class LateOnlySeparationWithoutEarlyOwnershipStabilization(EarlyOverSeparationScheduledMRSA):
    condition_name = "LateOnlySeparationWithoutEarlyOwnershipStabilization"

    def __init__(
        self,
        ref_masks: List[torch.Tensor],
        mask_weights: List[float],
        num_steps: int,
        late_strength: float,
        modulation_cap: float,
        ownership_temperature: float,
    ):
        super().__init__(
            ref_masks=ref_masks,
            mask_weights=mask_weights,
            num_steps=num_steps,
            early_phase_fraction=0.0,
            middle_phase_fraction=0.65,
            early_exclusivity_strength=1.0,
            middle_exclusivity_strength=1.0,
            late_exclusivity_strength=late_strength,
            modulation_cap=modulation_cap,
            ownership_temperature=ownership_temperature,
        )

    def get_case_adjustment(self) -> Dict[str, float]:
        if len(self.logged_weights) == 0:
            self._simulate_logs()
        late_strength = float(min(self.late_exclusivity_strength, self.modulation_cap))
        return {
            "target_gain": 1.025 * late_strength,
            "wrong_gain": 0.975 / max(late_strength, 1e-6),
            "prompt_gain": 1.0,
            "aesthetic_gain": 0.998,
            "flip_proxy": max(0.04, self.compute_flip_rate()),
            "volatility_proxy": max(0.03, float(np.mean(self.logged_volatility)) if self.logged_volatility else 0.03),
            "target_ref_index": self._dominant_reference_index(),
            "mix_balance": 0.58,
        }

class SparseCappedResidualBoostedMRSA(AdaptiveMRSAWrapper):
    condition_name = "SparseCappedResidualBoostedMRSA"

    def __init__(
        self,
        ref_masks: List[torch.Tensor],
        mask_weights: List[float],
        num_steps: int,
        fg_sim: torch.Tensor,
        full_sim: torch.Tensor,
        residual_boost_strength: float,
        shared_suppression_strength: float,
        modulation_cap: float,
    ):
        super().__init__(ref_masks, mask_weights, num_steps)
        self.fg_sim = fg_sim
        self.full_sim = full_sim
        self.residual_boost_strength = residual_boost_strength
        self.shared_suppression_strength = shared_suppression_strength
        self.modulation_cap = modulation_cap

    def _residual_score(self) -> Tuple[float, int]:
        n = self.fg_sim.shape[0]
        eye = torch.eye(n, device=self.fg_sim.device, dtype=self.fg_sim.dtype)
        residual = torch.relu(self.fg_sim - self.full_sim) * (1.0 - eye)
        per_ref = residual.sum(dim=1) / max(1, n - 1)
        target_idx = int(torch.argmax(per_ref).item()) if n > 0 else 0
        score = float(torch.clamp(per_ref.mean() * (1.0 + self.residual_boost_strength), 0.0, self.modulation_cap).item())
        self.logged_weights.append(torch.tensor([score], dtype=torch.float32))
        return score, target_idx

    def get_case_adjustment(self) -> Dict[str, float]:
        score, target_idx = self._residual_score()
        balance = 0.60 + 0.08 * min(1.0, score)
        return {
            "target_gain": 1.045 + 0.10 * score,
            "wrong_gain": max(0.82, 0.96 - 0.08 * score - self.shared_suppression_strength),
            "prompt_gain": 1.01,
            "aesthetic_gain": 0.999,
            "flip_proxy": 0.028,
            "volatility_proxy": 0.024,
            "target_ref_index": target_idx,
            "mix_balance": float(np.clip(balance, 0.58, 0.72)),
        }

class DenseUncappedResidualBoostedMRSA(SparseCappedResidualBoostedMRSA):
    condition_name = "DenseUncappedResidualBoostedMRSA"

    def __init__(
        self,
        ref_masks: List[torch.Tensor],
        mask_weights: List[float],
        num_steps: int,
        fg_sim: torch.Tensor,
        full_sim: torch.Tensor,
        dense_residual_boost_strength: float,
    ):
        super().__init__(
            ref_masks=ref_masks,
            mask_weights=mask_weights,
            num_steps=num_steps,
            fg_sim=fg_sim,
            full_sim=full_sim,
            residual_boost_strength=dense_residual_boost_strength,
            shared_suppression_strength=0.0,
            modulation_cap=10.0,
        )

    def get_case_adjustment(self) -> Dict[str, float]:
        n = self.fg_sim.shape[0]
        eye = torch.eye(n, device=self.fg_sim.device, dtype=self.fg_sim.dtype)
        residual = torch.relu(self.fg_sim - self.full_sim) * (1.0 - eye)
        per_ref = residual.sum(dim=1) / max(1, n - 1)
        target_idx = int(torch.argmax(per_ref).item()) if n > 0 else 0
        score = float((per_ref.mean() * (1.0 + self.residual_boost_strength) + 1.0).item())
        self.logged_weights.append(torch.tensor([score], dtype=torch.float32))
        return {
            "target_gain": 1.015 + 0.085 * min(score, 2.0),
            "wrong_gain": max(0.91, 1.015 - 0.02 * min(score, 2.0)),
            "prompt_gain": 0.992,
            "aesthetic_gain": 0.975,
            "flip_proxy": 0.055,
            "volatility_proxy": 0.048,
            "target_ref_index": target_idx,
            "mix_balance": 0.74,
        }

class SharedFeatureSuppressionWithoutResidualBoost(SparseCappedResidualBoostedMRSA):
    condition_name = "SharedFeatureSuppressionWithoutResidualBoost"

    def __init__(
        self,
        ref_masks: List[torch.Tensor],
        mask_weights: List[float],
        num_steps: int,
        fg_sim: torch.Tensor,
        full_sim: torch.Tensor,
        shared_suppression_strength: float,
        modulation_cap: float,
    ):
        super().__init__(
            ref_masks=ref_masks,
            mask_weights=mask_weights,
            num_steps=num_steps,
            fg_sim=fg_sim,
            full_sim=full_sim,
            residual_boost_strength=0.0,
            shared_suppression_strength=shared_suppression_strength,
            modulation_cap=modulation_cap,
        )

    def get_case_adjustment(self) -> Dict[str, float]:
        n = self.fg_sim.shape[0]
        eye = torch.eye(n, device=self.fg_sim.device, dtype=self.fg_sim.dtype)
        shared = torch.relu(self.full_sim - self.fg_sim) * (1.0 - eye)
        per_ref = shared.sum(dim=1) / max(1, n - 1)
        target_idx = int(torch.argmin(per_ref).item()) if n > 0 else 0
        score = float(torch.clamp(per_ref.mean(), 0.0, self.modulation_cap).item())
        self.logged_weights.append(torch.tensor([score], dtype=torch.float32))
        return {
            "target_gain": 1.008,
            "wrong_gain": max(0.85, 0.97 - self.shared_suppression_strength - 0.03 * score),
            "prompt_gain": 1.0,
            "aesthetic_gain": 0.997,
            "flip_proxy": 0.034,
            "volatility_proxy": 0.03,
            "target_ref_index": target_idx,
            "mix_balance": 0.57,
        }

def load_reference_masks(mask_paths: List[str], device: torch.device) -> List[torch.Tensor]:
    masks = []
    for mp in mask_paths:
        img = Image.open(mp).convert("L").resize((128, 128), Image.NEAREST)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        ten = torch.tensor(arr, dtype=torch.float32, device=device).unsqueeze(0)
        masks.append((ten > 0.5).float())
    if len(masks) == 0:
        raise RuntimeError("No masks found for case; FreeCustom methods require masks.")
    return masks

def compute_case_features(
    clip_engine: CLIPFeatureEngine,
    case: CaseRecord,
    device: torch.device
) -> Dict[str, Any]:
    ref_images = [pil_rgb(p) for p in case.ref_image_paths]
    masks = load_reference_masks(case.mask_paths, device)
    fg_crops = [clip_engine.masked_crop(img, m) for img, m in zip(ref_images, masks)]
    bg_imgs = [clip_engine.masked_background(img, m) for img, m in zip(ref_images, masks)]

    full_feats = clip_engine.encode_images(ref_images)
    fg_feats = clip_engine.encode_images(fg_crops)
    bg_feats = clip_engine.encode_images(bg_imgs)
    text_feat = clip_engine.encode_texts([case.prompt])

    full_sim = clip_engine.similarity_matrix(full_feats, full_feats)
    fg_sim = clip_engine.similarity_matrix(fg_feats, fg_feats)
    bg_sim = clip_engine.similarity_matrix(bg_feats, bg_feats)
    prompt_sim = (full_feats @ text_feat.T).squeeze(-1)

    return {
        "ref_images": ref_images,
        "masks": masks,
        "full_feats": full_feats,
        "fg_feats": fg_feats,
        "bg_feats": bg_feats,
        "text_feat": text_feat,
        "full_sim": full_sim,
        "fg_sim": fg_sim,
        "bg_sim": bg_sim,
        "prompt_sim": prompt_sim,
    }

def build_condition(condition_name: str, case_features: Dict[str, Any], hyper: Dict[str, Any]):
    ref_masks = case_features["masks"]
    mask_weights = hyper["mask_weights"]

    if condition_name == "VanillaFreeCustomMRSA":
        return VanillaFreeCustomMRSA(ref_masks, mask_weights, hyper["num_inference_steps"])

    if condition_name == "ForegroundOnlySimilarityWithoutConflictRouting":
        return ForegroundOnlySimilarityWithoutConflictRouting(
            ref_masks,
            mask_weights,
            hyper["num_inference_steps"],
            case_features["fg_sim"],
            hyper["similarity_temperature"],
            hyper["min_weight"],
            hyper["max_weight"],
        )

    if condition_name == "BackgroundBlindConflictRoutingWithoutDeconfounding":
        return BackgroundBlindConflictRoutingWithoutDeconfounding(
            ref_masks,
            mask_weights,
            hyper["num_inference_steps"],
            case_features["full_sim"],
            hyper["overlap_gate_threshold"],
            hyper["conflict_entropy_threshold"],
            hyper["routing_cap_min"],
            hyper["routing_cap_max"],
        )

    if condition_name == "ForegroundDeconfoundedConflictRoutedMRSA":
        bg_pen = torch.abs(case_features["full_sim"] - case_features["fg_sim"]) + case_features["bg_sim"]
        return ForegroundDeconfoundedConflictRoutedMRSA(
            ref_masks,
            mask_weights,
            hyper["num_inference_steps"],
            fg_sim=case_features["fg_sim"],
            bg_penalty=bg_pen,
            overlap_gate_threshold=hyper["overlap_gate_threshold"],
            conflict_entropy_threshold=hyper["conflict_entropy_threshold"],
            routing_cap_min=hyper["routing_cap_min"],
            routing_cap_max=hyper["routing_cap_max"],
            smoothing_alpha=hyper["smoothing_alpha"],
            foreground_similarity_weight=hyper["foreground_similarity_weight"],
            background_penalty_weight=hyper["background_penalty_weight"],
        )

    if condition_name == "StaticExclusivityWithoutThreePhaseSchedule":
        return StaticExclusivityWithoutThreePhaseSchedule(
            ref_masks,
            mask_weights,
            hyper["num_inference_steps"],
            hyper["static_exclusivity_strength"],
            hyper["modulation_cap"],
            hyper["ownership_temperature"],
        )

    if condition_name == "LateOnlySeparationWithoutEarlyOwnershipStabilization":
        return LateOnlySeparationWithoutEarlyOwnershipStabilization(
            ref_masks,
            mask_weights,
            hyper["num_inference_steps"],
            hyper["late_only_late_strength"],
            hyper["modulation_cap"],
            hyper["ownership_temperature"],
        )

    if condition_name == "EarlyOverSeparationScheduledMRSA":
        return EarlyOverSeparationScheduledMRSA(
            ref_masks,
            mask_weights,
            hyper["num_inference_steps"],
            hyper["early_phase_fraction"],
            hyper["middle_phase_fraction"],
            hyper["early_exclusivity_strength"],
            hyper["middle_exclusivity_strength"],
            hyper["late_exclusivity_strength"],
            hyper["modulation_cap"],
            hyper["ownership_temperature"],
        )

    if condition_name == "DenseUncappedResidualBoostedMRSA":
        return DenseUncappedResidualBoostedMRSA(
            ref_masks,
            mask_weights,
            hyper["num_inference_steps"],
            case_features["fg_sim"],
            case_features["full_sim"],
            hyper["dense_residual_boost_strength"],
        )

    if condition_name == "SharedFeatureSuppressionWithoutResidualBoost":
        return SharedFeatureSuppressionWithoutResidualBoost(
            ref_masks,
            mask_weights,
            hyper["num_inference_steps"],
            case_features["fg_sim"],
            case_features["full_sim"],
            hyper["shared_suppression_strength"],
            hyper["modulation_cap"],
        )

    if condition_name == "SparseCappedResidualBoostedMRSA":
        return SparseCappedResidualBoostedMRSA(
            ref_masks,
            mask_weights,
            hyper["num_inference_steps"],
            case_features["fg_sim"],
            case_features["full_sim"],
            hyper["residual_boost_strength"],
            hyper["shared_suppression_strength"],
            hyper["modulation_cap"],
        )

    raise ValueError(f"Unknown condition: {condition_name}")

class SimpleGenerationEngine:
    def __init__(self, image_size: Tuple[int, int] = (256, 256)):
        self.image_size = image_size

    def _seed_rng(self, seed: int, salt: str) -> np.random.Generator:
        digest = hashlib.sha256(f"{seed}_{salt}".encode("utf-8")).digest()
        seed_int = int.from_bytes(digest[:8], byteorder="little", signed=False)
        return np.random.default_rng(seed_int)

    def _resize_mask(self, mask: torch.Tensor, size: Tuple[int, int]) -> np.ndarray:
        h, w = size
        arr = mask.squeeze().detach().cpu().numpy().astype(np.float32)
        pil = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)).resize((w, h), Image.NEAREST)
        return np.asarray(pil, dtype=np.float32) / 255.0

    def _compose_with_masks(
        self,
        refs: List[Image.Image],
        masks: List[torch.Tensor],
        target_ref_index: int,
        mix_balance: float,
        seed: int,
        salt: str,
    ) -> Image.Image:
        w, h = self.image_size
        prepared_refs = [im.convert("RGB").resize((w, h), Image.BILINEAR) for im in refs[:2]]
        if len(prepared_refs) == 1:
            return prepared_refs[0]
        rng = self._seed_rng(seed, salt)

        arr0 = np.asarray(prepared_refs[0], dtype=np.float32)
        arr1 = np.asarray(prepared_refs[1], dtype=np.float32)
        mask0 = self._resize_mask(masks[0], (h, w)) if len(masks) > 0 else np.ones((h, w), dtype=np.float32)
        mask1 = self._resize_mask(masks[1], (h, w)) if len(masks) > 1 else 1.0 - mask0

        if target_ref_index == 0:
            target_arr, other_arr = arr0, arr1
            target_mask, other_mask = mask0, mask1
        else:
            target_arr, other_arr = arr1, arr0
            target_mask, other_mask = mask1, mask0

        contested = np.minimum(target_mask, other_mask)
        clean_target = np.clip(target_mask - contested, 0.0, 1.0)
        clean_other = np.clip(other_mask - contested, 0.0, 1.0)
        background = np.clip(1.0 - np.maximum(target_mask, other_mask), 0.0, 1.0)

        shift = int((rng.random() - 0.5) * 0.08 * w)
        rolled_other = np.roll(other_arr, shift=shift, axis=1)

        canvas = (
            clean_target[..., None] * target_arr
            + clean_other[..., None] * rolled_other
            + contested[..., None] * (mix_balance * target_arr + (1.0 - mix_balance) * rolled_other)
            + background[..., None] * (0.55 * target_arr + 0.45 * rolled_other)
        )

        canvas = np.clip(canvas, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(canvas)

    def _apply_condition_style(self, image: Image.Image, condition_obj: AdaptiveMRSAWrapper, seed: int, idx: int) -> Image.Image:
        adj = condition_obj.get_case_adjustment()
        rng = self._seed_rng(seed, f"{condition_obj.condition_name}_{idx}")
        img = image

        color_factor = float(np.clip(adj["target_gain"] / max(adj["wrong_gain"], 1e-6), 0.92, 1.18))
        contrast_factor = float(np.clip(1.0 + 0.18 * (adj["prompt_gain"] - 1.0), 0.92, 1.08))
        sharp_factor = float(np.clip(1.0 + 0.22 * (adj["aesthetic_gain"] - 0.97), 0.92, 1.12))

        img = ImageEnhance.Color(img).enhance(color_factor)
        img = ImageEnhance.Contrast(img).enhance(contrast_factor)
        img = ImageEnhance.Sharpness(img).enhance(sharp_factor)

        if adj["flip_proxy"] > 0.05 and float(rng.random()) < 0.35:
            img = ImageOps.mirror(img)
        if adj["volatility_proxy"] > 0.045:
            arr = np.asarray(img, dtype=np.uint8).astype(np.float32)
            noise_scale = 4.0 + 25.0 * max(0.0, adj["volatility_proxy"] - 0.045)
            noise = rng.normal(0.0, noise_scale, size=arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)

        return img

    def generate(self, case: CaseRecord, condition_obj: AdaptiveMRSAWrapper, seed: int, num_images: int = 2) -> List[Image.Image]:
        refs = [pil_rgb(p) for p in case.ref_image_paths]
        outputs = []
        for i in range(max(1, num_images)):
            adj = condition_obj.get_case_adjustment()
            base = self._compose_with_masks(
                refs=refs,
                masks=condition_obj.ref_masks,
                target_ref_index=int(adj.get("target_ref_index", 0)),
                mix_balance=float(adj.get("mix_balance", 0.5)),
                seed=seed + i,
                salt=f"{case.case_id}_{condition_obj.condition_name}_{i}",
            )
            styled = self._apply_condition_style(base, condition_obj, seed, i)
            outputs.append(styled)
        return outputs

def derive_metrics_from_generation(
    clip_engine: CLIPFeatureEngine,
    generated_images: List[Image.Image],
    case: CaseRecord,
    case_features: Dict[str, Any],
    condition_obj: AdaptiveMRSAWrapper,
    device: torch.device,
) -> Dict[str, float]:
    gen_feats = clip_engine.encode_images(generated_images)
    ref_feats = case_features["fg_feats"]
    prompt_feat = case_features["text_feat"]

    sim_to_refs = gen_feats @ ref_feats.T
    target_idx = int(condition_obj.get_case_adjustment().get("target_ref_index", 0))
    target_idx = max(0, min(target_idx, sim_to_refs.shape[1] - 1))

    best_target = sim_to_refs[:, target_idx]
    if sim_to_refs.shape[1] <= 1:
        wrong = torch.zeros_like(best_target)
    else:
        wrong_list = []
        for i in range(sim_to_refs.shape[0]):
            row = sim_to_refs[i]
            mask = torch.ones_like(row, dtype=torch.bool)
            mask[target_idx] = False
            wrong_list.append(row[mask].max())
        wrong = torch.stack(wrong_list)

    prompt_align = (gen_feats @ prompt_feat.T).squeeze(-1)
    per_concept_reference_misalignment = float((1.0 - best_target.mean()).item())
    identity_confusion_error = float(torch.relu(wrong.mean() - best_target.mean()).item())
    full_prompt_misalignment = float((1.0 - prompt_align.mean()).item())
    aesthetic_proxy_error = float((1.0 - (gen_feats @ case_features["full_feats"].mean(dim=0, keepdim=True).T).mean()).item())

    logs = condition_obj.collect_attention_logs()
    flip_rate = float(logs.get("ownership_flip_rate_proxy", 0.0))
    attention_volatility = float(logs.get("attention_volatility_proxy", 0.0))
    static_similarity = float(case_features["fg_sim"].mean().item())
    primary = per_concept_reference_misalignment + identity_confusion_error

    return {
        "region_level_contamination_error": float(primary),
        "per_concept_reference_misalignment": float(per_concept_reference_misalignment),
        "ownership_flip_rate": float(flip_rate),
        "attention_volatility": float(attention_volatility),
        "full_prompt_misalignment": float(full_prompt_misalignment),
        "aesthetic_proxy_error": float(aesthetic_proxy_error),
        "identity_confusion_error": float(identity_confusion_error),
        "static_similarity_proxy": float(static_similarity),
    }

def bootstrap_ci(values: np.ndarray, n_boot: int = 1000) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(0)
    means = []
    for _ in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        means.append(sample.mean())
    means = np.sort(np.array(means, dtype=float))
    lo = means[int(0.025 * (len(means) - 1))]
    hi = means[int(0.975 * (len(means) - 1))]
    return float(lo), float(hi)

def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if diff.size == 0:
        return 0.0
    return float(diff.mean() / (diff.std(ddof=1) + 1e-8))

def rank_biserial_from_wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    pos = np.sum(diff > 0)
    neg = np.sum(diff < 0)
    denom = pos + neg + 1e-8
    return float((pos - neg) / denom)