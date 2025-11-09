import argparse
import json
import os
import sys
import subprocess
from typing import Tuple, Dict, Optional, List
import numpy as np


class MLPClassifier:
    """
    NumPy-only Multi-Layer Perceptron for multi-class classification (softmax + cross-entropy).

    Features:
    - ReLU activations
    - Inverted Dropout regularization
    - L2 weight decay
    - Adam optimizer
    - He initialization
    """

    def __init__(
        self,
        layer_sizes: List[int],
        learning_rate: float = 1e-3,
        l2_lambda: float = 1e-4,
        dropout_rate: float = 0.0,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 2e-4,
        use_adamw: bool = True,
        grad_clip: Optional[float] = 5.0,
        use_ema: bool = True,
        ema_decay: float = 0.999,
        label_smoothing_eps: float = 0.0,
        class_weights: Optional[np.ndarray] = None,
        random_seed: Optional[int] = 42,
    ) -> None:
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must include input and output sizes")
        self.layer_sizes = list(layer_sizes)
        self.learning_rate = float(learning_rate)
        self.l2_lambda = float(l2_lambda)
        self.dropout_rate = float(dropout_rate)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.weight_decay = float(weight_decay)
        self.use_adamw = bool(use_adamw)
        self.grad_clip = None if grad_clip is None else float(grad_clip)
        self.use_ema = bool(use_ema)
        self.ema_decay = float(ema_decay)
        self.label_smoothing_eps = float(label_smoothing_eps)
        self.class_weights = class_weights

        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

        # Parameters
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []

        # Adam optimizer state
        self.m_w: List[np.ndarray] = []
        self.v_w: List[np.ndarray] = []
        self.m_b: List[np.ndarray] = []
        self.v_b: List[np.ndarray] = []
        self.step_count: int = 0

        self._initialize_parameters()

        # EMA buffers
        self.ema_w: List[np.ndarray] = [w.copy() for w in self.weights]
        self.ema_b: List[np.ndarray] = [b.copy() for b in self.biases]

    def _initialize_parameters(self) -> None:
        self.weights.clear()
        self.biases.clear()
        self.m_w.clear()
        self.v_w.clear()
        self.m_b.clear()
        self.v_b.clear()

        for in_dim, out_dim in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            # He initialization for ReLU layers
            w = self.rng.standard_normal((in_dim, out_dim), dtype=np.float32) * np.sqrt(2.0 / in_dim)
            b = np.zeros((out_dim,), dtype=np.float32)
            self.weights.append(w.astype(np.float32))
            self.biases.append(b.astype(np.float32))

            # Adam states
            self.m_w.append(np.zeros_like(w, dtype=np.float32))
            self.v_w.append(np.zeros_like(w, dtype=np.float32))
            self.m_b.append(np.zeros_like(b, dtype=np.float32))
            self.v_b.append(np.zeros_like(b, dtype=np.float32))

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0.0)

    @staticmethod
    def _relu_backward(grad_output: np.ndarray, pre_activation: np.ndarray) -> np.ndarray:
        mask = (pre_activation > 0).astype(grad_output.dtype)
        return grad_output * mask

    def _apply_dropout(self, activations: np.ndarray, dropout_rate: Optional[float] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        rate = dropout_rate if dropout_rate is not None else self.dropout_rate
        if rate <= 0.0:
            return activations, None
        keep_prob = 1.0 - rate
        mask = (self.rng.random(size=activations.shape, dtype=activations.dtype) < keep_prob).astype(activations.dtype)
        # Inverted dropout: scale during training so inference is a no-op
        dropped = activations * mask / keep_prob
        return dropped, mask

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        # Numerically stable softmax
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        return probs

    def forward(self, inputs: np.ndarray, training: bool = False, dropout_rate: Optional[float] = None) -> Tuple[np.ndarray, Dict[str, List[np.ndarray]]]:
        """
        Forward pass.
        Returns (logits, cache)
        cache contains intermediates for backprop: pre_activations, activations, dropout_masks
        """
        activations: List[np.ndarray] = [inputs]
        pre_activations: List[np.ndarray] = []
        dropout_masks: List[Optional[np.ndarray]] = []
        dropout_keep_probs: List[Optional[float]] = []

        a = inputs
        num_layers = len(self.weights)
        for layer_index in range(num_layers):
            w = self.weights[layer_index]
            b = self.biases[layer_index]
            z = a @ w + b  # pre-activation
            pre_activations.append(z)

            is_last = (layer_index == num_layers - 1)
            if not is_last:
                a = self._relu(z)
                if training and (dropout_rate if dropout_rate is not None else self.dropout_rate) > 0.0:
                    rate = (dropout_rate if dropout_rate is not None else self.dropout_rate)
                    a, mask = self._apply_dropout(a, rate)
                    keep_prob_used = 1.0 - rate
                else:
                    mask = None
                    keep_prob_used = None
                dropout_masks.append(mask)
                dropout_keep_probs.append(keep_prob_used)
                activations.append(a)
            else:
                # Last layer outputs logits (no activation)
                activations.append(z)
                dropout_masks.append(None)
                dropout_keep_probs.append(None)
                a = z

        cache = {
            "activations": activations,  # includes input and last logits
            "pre_activations": pre_activations,
            "dropout_masks": dropout_masks,
            "dropout_keep_probs": dropout_keep_probs,
        }
        return a, cache

    def compute_loss_and_gradients(self, inputs: np.ndarray, labels: np.ndarray, dropout_rate: Optional[float] = None) -> Tuple[float, List[np.ndarray], List[np.ndarray]]:
        """
        Compute softmax cross-entropy loss (with L2) and gradients via backprop.
        labels: integer class labels shape (N,)
        Returns: loss, dW_list, db_list
        """
        logits, cache = self.forward(inputs, training=True, dropout_rate=dropout_rate)

        # Softmax probabilities
        probs = self._softmax(logits)
        num_samples = inputs.shape[0]
        num_classes = probs.shape[1]
        # Cross-entropy loss (with optional label smoothing and class weights)
        if self.label_smoothing_eps > 0.0:
            eps = self.label_smoothing_eps
            y_soft = np.full((num_samples, num_classes), eps / (num_classes - 1), dtype=probs.dtype)
            y_soft[np.arange(num_samples), labels] = 1.0 - eps
            per_sample_loss = -np.sum(y_soft * np.log(probs + 1e-12), axis=1)
            dlogits = (probs - y_soft)
        else:
            correct_log_probs = -np.log(probs[np.arange(num_samples), labels] + 1e-12)
            per_sample_loss = correct_log_probs
            dlogits = probs
            dlogits[np.arange(num_samples), labels] -= 1.0

        # Apply class weights if provided
        if self.class_weights is not None:
            sw = self.class_weights[labels].astype(per_sample_loss.dtype)  # (N,)
            sum_w = float(np.sum(sw)) + 1e-12
            data_loss = float(np.sum(per_sample_loss * sw) / sum_w)
            dlogits *= (sw / sum_w)[:, None]
        else:
            data_loss = float(np.mean(per_sample_loss))
            dlogits /= float(num_samples)

        # L2 regularization (only on weights) - disabled when using AdamW
        l2_loss = 0.0
        if (not self.use_adamw) and (self.l2_lambda > 0.0):
            for w in self.weights:
                l2_loss += 0.5 * self.l2_lambda * float(np.sum(w * w))
        loss = data_loss + l2_loss

        # Gradients: use dlogits computed above (already weighted/normalized)

        dW_list: List[np.ndarray] = [np.zeros_like(w) for w in self.weights]
        db_list: List[np.ndarray] = [np.zeros_like(b) for b in self.biases]

        # Backprop last layer
        activations = cache["activations"]
        pre_activations = cache["pre_activations"]
        dropout_masks = cache["dropout_masks"]
        num_layers = len(self.weights)
        dropout_keep_probs = cache.get("dropout_keep_probs", [None] * (num_layers - 1) + [None])
        grad = dlogits  # gradient w.r.t. last layer pre-activation (logits)

        for layer_index in reversed(range(num_layers)):
            a_prev = activations[layer_index]  # activation from previous layer (or inputs for first layer)
            w = self.weights[layer_index]

            # Gradients for current layer weights/biases
            dW = a_prev.T @ grad
            db = np.sum(grad, axis=0)

            # Add L2 grad component on weights (only when not using AdamW)
            if (not self.use_adamw) and (self.l2_lambda > 0.0):
                dW += self.l2_lambda * w

            dW_list[layer_index] = dW.astype(np.float32)
            db_list[layer_index] = db.astype(np.float32)

            if layer_index > 0:
                # Propagate to previous layer
                grad = grad @ w.T
                # Undo dropout (inverted dropout scales in forward)
                mask = dropout_masks[layer_index - 1]
                if mask is not None:
                    keep_prob = dropout_keep_probs[layer_index - 1]
                    if keep_prob is None or keep_prob <= 0.0:
                        keep_prob = 1.0 - self.dropout_rate
                    grad = grad * mask / keep_prob
                # ReLU backward
                z_prev = pre_activations[layer_index - 1]
                grad = self._relu_backward(grad, z_prev)

        return float(loss), dW_list, db_list

    def apply_gradients(self, dW_list: List[np.ndarray], db_list: List[np.ndarray]) -> Tuple[float, float, float]:
        """
        Apply gradients and return metrics for logging.
        Returns: (grad_norm, step_size_ratio, weight_decay_magnitude)
        """
        self.step_count += 1
        lr, b1, b2, eps = self.learning_rate, self.beta1, self.beta2, self.epsilon
        t = self.step_count

        # Compute gradient norm before clipping
        grad_sq_sum = 0.0
        for dW, db in zip(dW_list, db_list):
            grad_sq_sum += float(np.sum(dW * dW) + np.sum(db * db))
        grad_norm = float(np.sqrt(grad_sq_sum + 1e-12))

        # Global norm gradient clipping
        if self.grad_clip is not None and self.grad_clip > 0.0:
            if grad_norm > self.grad_clip:
                scale = self.grad_clip / (grad_norm + 1e-12)
                dW_list = [dW * scale for dW in dW_list]
                db_list = [db * scale for db in db_list]

        # Track weight norms before update
        weight_norms_before = [float(np.linalg.norm(w)) for w in self.weights]
        
        # Track weight decay magnitude
        weight_decay_magnitude = 0.0
        if self.use_adamw and self.weight_decay > 0.0:
            for w in self.weights:
                weight_decay_magnitude += float(np.sum((lr * self.weight_decay * w) ** 2))
            weight_decay_magnitude = float(np.sqrt(weight_decay_magnitude))

        for i in range(len(self.weights)):
            dW, db = dW_list[i], db_list[i]

            # Update momentum terms
            self.m_w[i] = b1 * self.m_w[i] + (1.0 - b1) * dW
            self.v_w[i] = b2 * self.v_w[i] + (1.0 - b2) * (dW * dW)
            self.m_b[i] = b1 * self.m_b[i] + (1.0 - b1) * db
            self.v_b[i] = b2 * self.v_b[i] + (1.0 - b2) * (db * db)

            # Bias correction
            m_w_hat = self.m_w[i] / (1.0 - (b1 ** t))
            v_w_hat = self.v_w[i] / (1.0 - (b2 ** t))
            m_b_hat = self.m_b[i] / (1.0 - (b1 ** t))
            v_b_hat = self.v_b[i] / (1.0 - (b2 ** t))

            # Decoupled weight decay (AdamW)
            if self.use_adamw and self.weight_decay > 0.0:
                self.weights[i] -= lr * self.weight_decay * self.weights[i]

            # Parameter update (Adam)
            self.weights[i] -= lr * (m_w_hat / (np.sqrt(v_w_hat) + eps))
            self.biases[i] -= lr * (m_b_hat / (np.sqrt(v_b_hat) + eps))

            # EMA update
            if self.use_ema:
                self.ema_w[i] = self.ema_decay * self.ema_w[i] + (1.0 - self.ema_decay) * self.weights[i]
                self.ema_b[i] = self.ema_decay * self.ema_b[i] + (1.0 - self.ema_decay) * self.biases[i]

        # Compute step size ratio (||ΔW|| / ||W||)
        step_size_ratio = 0.0
        for i, (w_before, w_after) in enumerate(zip(weight_norms_before, [float(np.linalg.norm(w)) for w in self.weights])):
            if w_before > 1e-8:
                step_size_ratio += abs(w_after - w_before) / w_before
        step_size_ratio /= len(self.weights)

        return grad_norm, step_size_ratio, weight_decay_magnitude

    def swap_to_ema(self) -> None:
        if not self.use_ema or getattr(self, "_ema_swapped", False):
            return
        self._bak_w = [w.copy() for w in self.weights]
        self._bak_b = [b.copy() for b in self.biases]
        self.weights = [w.copy() for w in self.ema_w]
        self.biases = [b.copy() for b in self.ema_b]
        self._ema_swapped = True

    def swap_from_ema(self) -> None:
        if not self.use_ema or not getattr(self, "_ema_swapped", False):
            return
        self.weights, self.biases = self._bak_w, self._bak_b
        self._bak_w = self._bak_b = None
        self._ema_swapped = False

    def predict_proba(self, inputs: np.ndarray) -> np.ndarray:
        if self.use_ema:
            self.swap_to_ema()
            try:
                logits, _ = self.forward(inputs, training=False)
            finally:
                self.swap_from_ema()
        else:
            logits, _ = self.forward(inputs, training=False)
        probs = self._softmax(logits)
        return probs

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(inputs)
        return np.argmax(probs, axis=1)

    def predict_proba_mc(
        self,
        inputs: np.ndarray,
        n_samples: int = 20,
        temperature: float = 1.0,
        seed: Optional[int] = None,
        dropout_rate_override: Optional[float] = None,
    ) -> np.ndarray:
        """MC-Dropout: 推論時もdropoutをONにしてN回予測の平均"""
        # 乱数固定が必要なとき用
        rng_backup = None
        if seed is not None:
            rng_backup = self.rng
            self.rng = np.random.default_rng(seed)

        if self.use_ema:
            self.swap_to_ema()
        try:
            N = inputs.shape[0]
            C = self.layer_sizes[-1]
            acc = np.zeros((N, C), dtype=np.float32)
            for _ in range(n_samples):
                logits, _ = self.forward(inputs, training=True, dropout_rate=dropout_rate_override)  # ← dropout有効！
                logits = logits / max(temperature, 1e-6)
                logits = logits - logits.max(axis=1, keepdims=True)
                p = np.exp(logits)
                acc += p / p.sum(axis=1, keepdims=True)
            acc /= float(n_samples)
            return acc
        finally:
            if self.use_ema:
                self.swap_from_ema()
            if rng_backup is not None:
                self.rng = rng_backup

    def get_state(self) -> Dict[str, List[np.ndarray]]:
        return {
            "weights": [w.copy() for w in self.weights],
            "biases": [b.copy() for b in self.biases],
            "m_w": [m.copy() for m in self.m_w],
            "v_w": [v.copy() for v in self.v_w],
            "m_b": [m.copy() for m in self.m_b],
            "v_b": [v.copy() for v in self.v_b],
            "step_count": self.step_count,
            "ema_w": [w.copy() for w in self.ema_w],
            "ema_b": [b.copy() for b in self.ema_b],
        }

    def set_state(self, state: Dict[str, List[np.ndarray]]) -> None:
        self.weights = [w.copy() for w in state["weights"]]
        self.biases = [b.copy() for b in state["biases"]]
        self.m_w = [m.copy() for m in state.get("m_w", [np.zeros_like(w) for w in self.weights])]
        self.v_w = [v.copy() for v in state.get("v_w", [np.zeros_like(w) for w in self.weights])]
        self.m_b = [m.copy() for m in state.get("m_b", [np.zeros_like(b) for b in self.biases])]
        self.v_b = [v.copy() for v in state.get("v_b", [np.zeros_like(b) for b in self.biases])]
        self.step_count = int(state.get("step_count", 0))
        self.ema_w = [w.copy() for w in state.get("ema_w", [p.copy() for p in self.weights])]
        self.ema_b = [b.copy() for b in state.get("ema_b", [p.copy() for p in self.biases])]

    def save_npz(self, path: str, standardization_mean: Optional[np.ndarray] = None, standardization_std: Optional[np.ndarray] = None, meta: Optional[Dict] = None) -> None:
        data = {
            "layer_sizes": np.array(self.layer_sizes, dtype=np.int32),
            "learning_rate": np.array(self.learning_rate, dtype=np.float32),
            "l2_lambda": np.array(self.l2_lambda, dtype=np.float32),
            "dropout_rate": np.array(self.dropout_rate, dtype=np.float32),
            "beta1": np.array(self.beta1, dtype=np.float32),
            "beta2": np.array(self.beta2, dtype=np.float32),
            "epsilon": np.array(self.epsilon, dtype=np.float32),
            "weight_decay": np.array(self.weight_decay, dtype=np.float32),
            "use_adamw": np.array(1 if self.use_adamw else 0, dtype=np.int8),
            "grad_clip": np.array(-1.0 if self.grad_clip is None else float(self.grad_clip), dtype=np.float32),
            "use_ema": np.array(1 if self.use_ema else 0, dtype=np.int8),
            "ema_decay": np.array(self.ema_decay, dtype=np.float32),
            "label_smoothing_eps": np.array(self.label_smoothing_eps, dtype=np.float32),
            "random_seed": np.array(self.random_seed if self.random_seed is not None else -1, dtype=np.int32),
        }

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            data[f"W_{i}"] = w
            data[f"b_{i}"] = b
            # EMA parameters
            data[f"EMA_W_{i}"] = self.ema_w[i]
            data[f"EMA_b_{i}"] = self.ema_b[i]

        if standardization_mean is not None:
            data["standardization_mean"] = standardization_mean.astype(np.float32)
        if standardization_std is not None:
            data["standardization_std"] = standardization_std.astype(np.float32)

        if meta is not None:
            # Store JSON as bytes to keep inside npz
            meta_json = json.dumps(meta, ensure_ascii=False).encode("utf-8")
            data["meta_json"] = np.frombuffer(meta_json, dtype=np.uint8)

        np.savez(path, **data)

    @staticmethod
    def load_npz(path: str) -> "MLPClassifier":
        npz = np.load(path, allow_pickle=False)
        layer_sizes = npz["layer_sizes"].astype(int).tolist()
        files = set(npz.files)
        learning_rate = float(npz["learning_rate"]) if "learning_rate" in files else 1e-3
        l2_lambda = float(npz["l2_lambda"]) if "l2_lambda" in files else 1e-4
        dropout_rate = float(npz["dropout_rate"]) if "dropout_rate" in files else 0.0
        beta1 = float(npz["beta1"]) if "beta1" in files else 0.9
        beta2 = float(npz["beta2"]) if "beta2" in files else 0.999
        epsilon = float(npz["epsilon"]) if "epsilon" in files else 1e-8
        weight_decay = float(npz["weight_decay"]) if "weight_decay" in files else 2e-4
        use_adamw = bool(npz["use_adamw"]) if "use_adamw" in files else True
        grad_clip_val = float(npz["grad_clip"]) if "grad_clip" in files else 5.0
        grad_clip = None if grad_clip_val < 0.0 else grad_clip_val
        use_ema = bool(npz["use_ema"]) if "use_ema" in files else True
        ema_decay = float(npz["ema_decay"]) if "ema_decay" in files else 0.999
        label_smoothing_eps = float(npz["label_smoothing_eps"]) if "label_smoothing_eps" in files else 0.0
        random_seed = int(npz["random_seed"]) if "random_seed" in files else 42

        model = MLPClassifier(
            layer_sizes=layer_sizes,
            learning_rate=learning_rate,
            l2_lambda=l2_lambda,
            dropout_rate=dropout_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            weight_decay=weight_decay,
            use_adamw=use_adamw,
            grad_clip=grad_clip,
            use_ema=use_ema,
            ema_decay=ema_decay,
            label_smoothing_eps=label_smoothing_eps,
            random_seed=random_seed,
        )
        weights: List[np.ndarray] = []
        biases: List[np.ndarray] = []
        for i in range(len(layer_sizes) - 1):
            weights.append(npz[f"W_{i}"])  # type: ignore[index]
            biases.append(npz[f"b_{i}"])   # type: ignore[index]
        model.weights = weights
        model.biases = biases
        # Load EMA params if present
        ema_w: List[np.ndarray] = []
        ema_b: List[np.ndarray] = []
        has_ema = all((f"EMA_W_{i}" in files and f"EMA_b_{i}" in files) for i in range(len(layer_sizes) - 1))
        if has_ema:
            for i in range(len(layer_sizes) - 1):
                ema_w.append(npz[f"EMA_W_{i}"])  # type: ignore[index]
                ema_b.append(npz[f"EMA_b_{i}"])  # type: ignore[index]
            model.ema_w = ema_w
            model.ema_b = ema_b
        else:
            model.ema_w = [w.copy() for w in model.weights]
            model.ema_b = [b.copy() for b in model.biases]
        return model

    def get_activation_stats(self, inputs: np.ndarray) -> Dict[str, List[float]]:
        """Get ReLU activation statistics for each layer."""
        # Manually compute forward pass to get ReLU activations
        a = inputs
        stats = {"zero_rates": [], "mean_acts": [], "std_acts": []}
        
        for i in range(len(self.weights) - 1):  # All layers except last (output)
            w = self.weights[i]
            b = self.biases[i]
            z = a @ w + b  # pre-activation
            a = self._relu(z)  # ReLU activation
            
            zero_rate = float(np.mean(a == 0.0))
            mean_act = float(np.mean(a))
            std_act = float(np.std(a))
            stats["zero_rates"].append(zero_rate)
            stats["mean_acts"].append(mean_act)
            stats["std_acts"].append(std_act)
        
        return stats

    def get_weight_norms(self) -> Dict[str, List[float]]:
        """Get L2 norms of weights and biases for each layer."""
        weight_norms = [float(np.linalg.norm(w)) for w in self.weights]
        bias_norms = [float(np.linalg.norm(b)) for b in self.biases]
        return {"weight_norms": weight_norms, "bias_norms": bias_norms}

    def check_nan_inf(self) -> Dict[str, bool]:
        """Check for NaN/Inf in weights, biases, and optimizer states."""
        has_nan_inf = {
            "weights": any(np.any(~np.isfinite(w)) for w in self.weights),
            "biases": any(np.any(~np.isfinite(b)) for b in self.biases),
            "m_w": any(np.any(~np.isfinite(m)) for m in self.m_w),
            "v_w": any(np.any(~np.isfinite(v)) for v in self.v_w),
            "m_b": any(np.any(~np.isfinite(m)) for m in self.m_b),
            "v_b": any(np.any(~np.isfinite(v)) for v in self.v_b),
        }
        return has_nan_inf

    @staticmethod
    def accuracy(predicted_labels: np.ndarray, true_labels: np.ndarray) -> float:
        return float(np.mean(predicted_labels == true_labels))

    @staticmethod
    def confusion_matrix(predicted_labels: np.ndarray, true_labels: np.ndarray, num_classes: int) -> np.ndarray:
        """Compute confusion matrix."""
        cm = np.zeros((num_classes, num_classes), dtype=np.int32)
        for true, pred in zip(true_labels, predicted_labels):
            cm[true, pred] += 1
        return cm

    @staticmethod
    def per_class_metrics(predicted_labels: np.ndarray, true_labels: np.ndarray, num_classes: int) -> Dict[str, List[float]]:
        """Compute precision, recall, F1 for each class."""
        cm = MLPClassifier.confusion_matrix(predicted_labels, true_labels, num_classes)
        precision = []
        recall = []
        f1 = []
        
        for i in range(num_classes):
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            fn = np.sum(cm[i, :]) - tp
            
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            
            precision.append(p)
            recall.append(r)
            f1.append(f)
        
        return {"precision": precision, "recall": recall, "f1": f1}

    @staticmethod
    def top_k_errors(predicted_labels: np.ndarray, true_labels: np.ndarray, losses: np.ndarray, k: int = 5) -> List[Dict]:
        """Get top-k highest loss errors."""
        error_indices = np.argsort(losses)[-k:][::-1]
        errors = []
        for idx in error_indices:
            errors.append({
                "index": int(idx),
                "true": int(true_labels[idx]),
                "pred": int(predicted_labels[idx]),
                "loss": float(losses[idx])
            })
        return errors

    @staticmethod
    def calibration_metrics(probs: np.ndarray, true_labels: np.ndarray, num_bins: int = 10) -> Dict[str, float]:
        """Compute calibration metrics: ECE, entropy, margin."""
        predicted_labels = np.argmax(probs, axis=1)
        
        # Expected Calibration Error
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probs.max(axis=1) > bin_lower) & (probs.max(axis=1) <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = (predicted_labels[in_bin] == true_labels[in_bin]).mean()
                avg_confidence_in_bin = probs[in_bin].max(axis=1).mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        # Entropy and margin
        entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)
        max_probs = probs[np.arange(len(probs)), predicted_labels]
        second_max_probs = np.partition(probs, -2, axis=1)[:, -2]
        margin = max_probs - second_max_probs
        
        return {
            "ece": float(ece),
            "mean_entropy": float(np.mean(entropy)),
            "mean_margin": float(np.mean(margin))
        }


def load_data(
    data_dir: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_train_path = os.path.join(data_dir, "input", "x_train.npy")
    y_train_path = os.path.join(data_dir, "input", "y_train.npy")
    x_test_path = os.path.join(data_dir, "input", "x_test.npy")

    x_train = np.load(x_train_path)
    y_train = np.load(y_train_path)
    x_test = np.load(x_test_path)

    if x_train.ndim > 2:
        x_train = x_train.reshape(x_train.shape[0], -1)
    if x_test.ndim > 2:
        x_test = x_test.reshape(x_test.shape[0], -1)

    y_train = y_train.astype(np.int64).reshape(-1)

    return x_train.astype(np.float32), y_train, x_test.astype(np.float32)


def standardize(
    x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-7
    return (x_train - mean) / std, (x_val - mean) / std, (x_test - mean) / std, mean.squeeze(), std.squeeze()


def train_valid_split(
    x: np.ndarray,
    y: np.ndarray,
    valid_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(n * valid_ratio)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    return x[tr_idx], y[tr_idx], x[val_idx], y[val_idx]


def iterate_minibatches(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True, seed: Optional[int] = None):
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    indices = np.arange(n)
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_idx = indices[start:end]
        yield x[batch_idx], y[batch_idx]


class CosineWithWarmup:
    def __init__(self, warmup_steps: int, total_steps: int, base_lr: float, min_lr: float = 1e-5):
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.base_lr = float(base_lr)
        self.min_lr = float(min_lr)

    def lr_at(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.base_lr * float(step + 1) / max(1, self.warmup_steps)
        progress = float(step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine


def _softmax_stable(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax with explicit max subtraction."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _build_base_cli_for_seeds(argv: List[str]) -> List[str]:
    """
    Extract CLI tokens that should be forwarded to per-seed subprocesses.
    Removes multi-seed orchestration flags and overrides that we will set per run.
    """
    base: List[str] = []
    i = 1
    argc = len(argv)
    while i < argc:
        tok = argv[i]
        if tok == "--multi_seed":
            i += 1
            while i < argc and not argv[i].startswith("--"):
                i += 1
            continue
        if tok in ("--multi_seed_skip_existing", "--preset_best", "--save_test_proba"):
            i += 1
            continue
        if tok in ("--seed", "--proba_path"):
            i += 2
            continue
        base.append(tok)
        i += 1
    return base


def _run_multi_seed_pipeline(
    seeds: List[int],
    base_cli: List[str],
    output_dir: str,
    skip_existing: bool,
    ensemble_weighted: bool,
    ensemble_weighted_power: float,
    ensemble_save_proba: bool,
    ensemble_save_proba_path: Optional[str],
) -> None:
    """Execute training for each seed and ensemble the results."""
    script_path = os.path.abspath(__file__)
    os.makedirs(output_dir, exist_ok=True)
    unique_seeds = list(dict.fromkeys(int(s) for s in seeds))
    proba_files: List[str] = []
    for seed in unique_seeds:
        proba_out = os.path.join(output_dir, f"proba_seed{seed}.npy")
        proba_files.append(proba_out)
        if skip_existing and os.path.exists(proba_out):
            print(f"[MultiSeed] Skip training seed={seed} (exists: {proba_out})")
            continue
        cmd = [
            sys.executable,
            script_path,
            *base_cli,
            "--seed",
            str(seed),
            "--save_test_proba",
            "--proba_path",
            proba_out,
        ]
        print(f"[MultiSeed] Running seed={seed}")
        subprocess.run(cmd, check=True)

    available = [p for p in proba_files if os.path.exists(p)]
    if len(available) == 0:
        print("[MultiSeed] No probability files found; skipping ensemble.")
        return

    cmd_ens = [sys.executable, script_path, "--ensemble_from", *available]
    if ensemble_weighted:
        cmd_ens.append("--ensemble_weighted")
        cmd_ens.extend(["--ensemble_weighted_power", str(ensemble_weighted_power)])
    if ensemble_save_proba:
        cmd_ens.append("--ensemble_save_proba")
        if ensemble_save_proba_path:
            cmd_ens.extend(["--ensemble_save_proba_path", ensemble_save_proba_path])
    subprocess.run(cmd_ens, check=True)
    print(f"[MultiSeed] Done. Wrote ensemble for seeds {unique_seeds}")


def tta_predict_proba_5dir_logits_mc(
    model: MLPClassifier,
    x_flat: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    temperature: float = 1.0,
    n_samples_total: int = 30,
    seed: Optional[int] = None,
    dropout_rate_override: Optional[float] = None,
) -> np.ndarray:
    """MC×TTA (5-direction, logit averaging, total 30 samples)"""
    shifts = [(0,0),(1,0),(-1,0),(0,1),(0,-1)]
    H = W = 28
    N, D = x_flat.shape
    C = model.layer_sizes[-1]
    imgs = x_flat.reshape(N, H, W)
    num_shifts = len(shifts)
    total_samples = max(1, int(n_samples_total))
    counts = np.full(num_shifts, total_samples // num_shifts, dtype=int)
    remainder = total_samples - counts.sum()
    if remainder > 0:
        counts[:remainder] += 1

    acc_logits = np.zeros((N, C), dtype=np.float32)
    rng_backup = None
    if seed is not None:
        rng_backup = model.rng
        model.rng = np.random.default_rng(seed)

    shifts_used = 0
    if model.use_ema:
        model.swap_to_ema()
    try:
        for idx, (dx, dy) in enumerate(shifts):
            samples_this = int(counts[idx])
            if samples_this <= 0:
                continue
            rolled = np.roll(np.roll(imgs, dy, axis=1), dx, axis=2)
            if dy > 0:   rolled[:, :dy, :] = 0
            elif dy < 0: rolled[:, dy:, :] = 0
            if dx > 0:   rolled[:, :, :dx] = 0
            elif dx < 0: rolled[:, :, dx:] = 0

            flat = (rolled.reshape(N, D) - mean) / std

            # MC: forward(training=True) を per_shift 回
            logits_sum = np.zeros((N, C), dtype=np.float32)
            for _ in range(samples_this):
                logits, _ = model.forward(flat, training=True, dropout_rate=dropout_rate_override)
                logits_sum += (logits / max(temperature, 1e-6)).astype(np.float32)
            acc_logits += logits_sum / float(samples_this)
            shifts_used += 1
    finally:
        if model.use_ema:
            model.swap_from_ema()
        if rng_backup is not None:
            model.rng = rng_backup

    # 方向平均 → 安定softmax
    denom = float(max(1, shifts_used))
    logits = acc_logits / denom
    return _softmax_stable(logits)


def tta_predict_proba_9dir_logits_mc(
    model: MLPClassifier,
    x_flat: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    temperature: float = 1.0,
    n_samples_total: int = 30,
    seed: Optional[int] = None,
    dropout_rate_override: Optional[float] = None,
) -> np.ndarray:
    """MC×TTA (9-direction, logit averaging)."""
    shifts = [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
    H = W = 28
    N, D = x_flat.shape
    C = model.layer_sizes[-1]
    imgs = x_flat.reshape(N, H, W)
    num_shifts = len(shifts)
    total_samples = max(1, int(n_samples_total))
    counts = np.full(num_shifts, total_samples // num_shifts, dtype=int)
    remainder = total_samples - counts.sum()
    if remainder > 0:
        counts[:remainder] += 1

    acc_logits = np.zeros((N, C), dtype=np.float32)
    rng_backup = None
    if seed is not None:
        rng_backup = model.rng
        model.rng = np.random.default_rng(seed)

    shifts_used = 0
    if model.use_ema:
        model.swap_to_ema()
    try:
        for idx, (dx, dy) in enumerate(shifts):
            samples_this = int(counts[idx])
            if samples_this <= 0:
                continue
            rolled = np.roll(np.roll(imgs, dy, axis=1), dx, axis=2)
            if dy > 0:   rolled[:, :dy, :] = 0
            elif dy < 0: rolled[:, dy:, :] = 0
            if dx > 0:   rolled[:, :, :dx] = 0
            elif dx < 0: rolled[:, :, dx:] = 0

            flat = (rolled.reshape(N, D) - mean) / std

            logits_sum = np.zeros((N, C), dtype=np.float32)
            for _ in range(samples_this):
                logits, _ = model.forward(flat, training=True, dropout_rate=dropout_rate_override)
                logits_sum += (logits / max(temperature, 1e-6)).astype(np.float32)
            acc_logits += logits_sum / float(samples_this)
            shifts_used += 1
    finally:
        if model.use_ema:
            model.swap_from_ema()
        if rng_backup is not None:
            model.rng = rng_backup

    logits = acc_logits / float(max(1, shifts_used))
    return _softmax_stable(logits)


def shift_batch_flat(xb_flat: np.ndarray, img_hw=(28, 28), max_shift: int = 2, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    N, D = xb_flat.shape
    H, W = img_hw
    xb = xb_flat.reshape(N, H, W)
    out = np.empty_like(xb)
    for i in range(N):
        dx = int(rng.integers(-max_shift, max_shift + 1))
        dy = int(rng.integers(-max_shift, max_shift + 1))
        img = xb[i]
        shifted = np.roll(np.roll(img, dy, axis=0), dx, axis=1)
        if dy > 0:
            shifted[:dy, :] = 0
        elif dy < 0:
            shifted[dy:, :] = 0
        if dx > 0:
            shifted[:, :dx] = 0
        elif dx < 0:
            shifted[:, dx:] = 0
        out[i] = shifted
    return out.reshape(N, D)

def cutout_batch_flat(xb_flat: np.ndarray, img_hw=(28, 28), max_size: int = 4, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Apply small Cutout augmentation to a batch of flattened images."""
    if rng is None:
        rng = np.random.default_rng()
    N, D = xb_flat.shape
    H, W = img_hw
    xb = xb_flat.reshape(N, H, W).copy()
    for i in range(N):
        s = int(rng.integers(2, max_size + 1))
        cy = int(rng.integers(0, H))
        cx = int(rng.integers(0, W))
        y0, y1 = max(0, cy - s // 2), min(H, cy + s // 2)
        x0, x1 = max(0, cx - s // 2), min(W, cx + s // 2)
        xb[i, y0:y1, x0:x1] = 0
    return xb.reshape(N, D)

def gaussian_noise_batch_flat(xb_flat: np.ndarray, sigma: float = 0.02, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Add small Gaussian noise to a batch of flattened images (on raw scale)."""
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0.0, float(sigma), size=xb_flat.shape).astype(xb_flat.dtype)
    return xb_flat + noise

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))
    parser.add_argument("--output_dir", type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "output"))
    parser.add_argument("--model_dir", type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "train"))

    parser.add_argument("--hidden_sizes", type=int, nargs="*", default=[512, 256, 128])
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--adamw", action="store_true", default=True, help="Use AdamW (decoupled weight decay)")
    parser.add_argument("--no-adamw", dest="adamw", action="store_false", help="Disable AdamW; use Adam instead")
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--ema", action="store_true", default=True, help="Use EMA parameters for inference")
    parser.add_argument("--no-ema", dest="ema", action="store_false", help="Disable EMA")
    parser.add_argument("--ema_decay", type=float, default=0.9995)
    parser.add_argument("--label_smoothing_eps", type=float, default=0.03)
    parser.add_argument("--mc_samples_val", type=int, default=10)
    parser.add_argument("--mc_samples_test", type=int, default=30)
    parser.add_argument("--mc_temp", type=float, default=1.0)
    parser.add_argument("--mc_dropout_override", type=float, default=None)
    parser.add_argument("--enable_noise_aug", action="store_true", default=False, help="Enable small Gaussian noise augmentation")
    parser.add_argument("--enable_adaptive_blend", action="store_true", default=False, help="Enable adaptive alpha search for 2-way blend")
    parser.add_argument("--enable_temp_scale", action="store_true", default=False, help="Enable post-hoc temperature scaling on selected candidate")
    parser.add_argument("--enable_swa", action="store_true", default=False, help="Enable SWA over EMA weights late in training")
    parser.add_argument("--swa_start_frac", type=float, default=0.8, help="Start SWA after this fraction of epochs")
    parser.add_argument("--swa_interval", type=int, default=2, help="Snapshot interval (epochs) for SWA")
    parser.add_argument("--class_weights_json", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    # Fine-tune and late-epoch controls
    parser.add_argument("--ft_epochs", type=int, default=2, help="Fine-tune epochs on full train at the end (0/1/2)")
    parser.add_argument("--final_dropout_target", type=float, default=0.05, help="Late-stage linear dropout target (after 80% epochs)")
    parser.add_argument("--save_test_proba", action="store_true", default=False, help="Save test probabilities to file")
    parser.add_argument("--proba_path", type=str, default=None, help="Path to save test probabilities (.npy)")
    parser.add_argument("--ensemble_from", type=str, nargs="*", default=None, help="Average given proba files and write y_pred.csv, then exit")
    parser.add_argument("--force_infer", type=str, default=None, help="Force inference mode: det | mc@T | tta5mc@T | tta9mc@T (e.g., mc@0.80)")
    parser.add_argument("--preset_best", action="store_true", default=False, help="Train seeds 42/43/44 with tuned params and ensemble")
    parser.add_argument("--ensemble_weighted", action="store_true", default=False, help="Weighted ensemble using sidecar meta JSON")
    parser.add_argument("--ensemble_weighted_power", type=float, default=1.0, help="Exponent for weights (w = acc^power)")
    parser.add_argument("--ensemble_save_proba", action="store_true", default=False, help="Also save the ensembled average probabilities as .npy")
    parser.add_argument("--ensemble_save_proba_path", type=str, default=None, help="Path to save averaged ensemble probabilities (.npy)")
    parser.add_argument("--multi_seed", type=int, nargs="+", default=None, help="Run multiple seeds sequentially (e.g., --multi_seed 42 43 44)")
    parser.add_argument("--multi_seed_skip_existing", action="store_true", default=False, help="Skip training a seed when its probability file already exists")

    args = parser.parse_args()

    # Ensemble mode: load probas, average, write y_pred.csv (and optionally avg proba) and exit
    if args.ensemble_from and len(args.ensemble_from) > 0:
        paths = list(args.ensemble_from)
        probas = [np.load(p) for p in paths]
        if args.ensemble_weighted:
            weights = []
            for p in paths:
                w = 1.0
                meta_path = p + ".meta.json"
                if os.path.exists(meta_path):
                    with open(meta_path, "r", encoding="utf-8") as f:
                        try:
                            md = json.load(f)
                            acc = float(md.get("selector_val_acc", 1.0))
                            w = max(1e-6, acc) ** float(args.ensemble_weighted_power)
                        except Exception:
                            w = 1.0
                weights.append(w)
            wsum = float(np.sum(weights)) if len(weights) > 0 else 1.0
            acc = np.zeros_like(probas[0], dtype=np.float32)
            for p_arr, w in zip(probas, weights):
                acc += (p_arr.astype(np.float32) * float(w))
            avg = acc / max(wsum, 1e-12)
        else:
            avg = np.mean(probas, axis=0)
        y_pred = np.argmax(avg, axis=1)
        pred_path = os.path.join(args.output_dir, "y_pred.csv")
        np.savetxt(pred_path, y_pred.astype(np.int64), fmt="%d", delimiter=",", newline="\n", header="label", comments="")
        if args.ensemble_save_proba:
            outp = args.ensemble_save_proba_path if args.ensemble_save_proba_path else os.path.join(args.output_dir, "ensemble_avg.npy")
            np.save(outp, avg.astype(np.float32))
            print(f"Saved ensemble averaged probabilities to {outp}")
        print(f"Ensembled {len(probas)} files -> {pred_path}")
        return

    if args.multi_seed:
        base_cli = _build_base_cli_for_seeds(sys.argv)
        _run_multi_seed_pipeline(
            seeds=list(args.multi_seed),
            base_cli=base_cli,
            output_dir=args.output_dir,
            skip_existing=bool(args.multi_seed_skip_existing),
            ensemble_weighted=bool(args.ensemble_weighted),
            ensemble_weighted_power=float(args.ensemble_weighted_power),
            ensemble_save_proba=bool(args.ensemble_save_proba),
            ensemble_save_proba_path=args.ensemble_save_proba_path,
        )
        return

    # Preset: best pipeline to reproduce LB ~0.92
    if args.preset_best:
        base_cli = _build_base_cli_for_seeds(sys.argv)
        base_cli.extend([
            "--hidden_sizes", "640", "320", "160",
            "--dropout", "0.12",
            "--weight_decay", "8e-05",
            "--class_weights_json", '{"6":1.06}',
            "--epochs", str(args.epochs),
            "--patience", str(args.patience),
        ])
        _run_multi_seed_pipeline(
            seeds=[42, 43, 44],
            base_cli=base_cli,
            output_dir=args.output_dir,
            skip_existing=True,
            ensemble_weighted=bool(args.ensemble_weighted),
            ensemble_weighted_power=float(args.ensemble_weighted_power),
            ensemble_save_proba=bool(args.ensemble_save_proba),
            ensemble_save_proba_path=args.ensemble_save_proba_path,
        )
        print("[PresetBest] Done. Wrote ensembled y_pred.csv")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    x_train, y_train, x_test = load_data(args.data_dir)

    input_dim = x_train.shape[1]
    num_classes = int(y_train.max()) + 1
    layer_sizes = [input_dim] + list(args.hidden_sizes) + [num_classes]

    x_tr, y_tr, x_val, y_val = train_valid_split(x_train, y_train, valid_ratio=0.1, seed=args.seed)
    x_tr, x_val, x_test_std, mean, std = standardize(x_tr, x_val, x_test)
    mean = mean.astype(np.float32)
    std = std.astype(np.float32)

    # Parse class weights if provided
    cw = None
    if args.class_weights_json:
        obj = json.loads(args.class_weights_json)
        cw = np.ones((num_classes,), dtype=np.float32)
        for k, v in obj.items():
            cw[int(k)] = float(v)

    use_adamw = bool(args.adamw)
    l2_lambda = 0.0 if use_adamw else args.l2

    model = MLPClassifier(
        layer_sizes=layer_sizes,
        learning_rate=args.lr,
        l2_lambda=l2_lambda,
        dropout_rate=args.dropout,
        weight_decay=args.weight_decay,
        use_adamw=use_adamw,
        grad_clip=None if args.grad_clip is None or args.grad_clip < 0 else float(args.grad_clip),
        use_ema=bool(args.ema),
        ema_decay=args.ema_decay,
        label_smoothing_eps=args.label_smoothing_eps,
        class_weights=cw,
        random_seed=args.seed,
    )

    best_val_acc = 0.0
    best_state = None
    epochs_no_improve = 0
    best_epoch = 0

    # Cosine + Warmup scheduler
    total_steps = int(np.ceil(x_tr.shape[0] / args.batch_size)) * args.epochs
    sched = CosineWithWarmup(
        warmup_steps=int(0.05 * total_steps),
        total_steps=total_steps,
        base_lr=args.lr,
        min_lr=max(1e-5, args.lr * 1e-2),
    )
    step = 0
    num_batches = int(np.ceil(x_tr.shape[0] / args.batch_size))

    # Augmentation tracking
    aug_applied_count = 0

    swa_ema_snapshots_w: List[List[np.ndarray]] = []
    swa_ema_snapshots_b: List[List[np.ndarray]] = []

    for epoch in range(1, args.epochs + 1):
        # Epoch-specific RNG for augmentation diversity (deterministic but different per epoch)
        rng_aug_epoch = np.random.default_rng(args.seed + epoch * 10007)
        
        # 終盤Dropout調整（80%以降で線形に最終ターゲットまで低減）
        if epoch > int(0.8 * args.epochs):
            frac = (epoch - int(0.8 * args.epochs)) / max(1, args.epochs - int(0.8 * args.epochs))
            target = float(args.final_dropout_target)
            current_dropout = max(0.0, args.dropout + (target - args.dropout) * frac)
        else:
            current_dropout = args.dropout
        
        # 末期でpを下げる（80%以降で0.7→0.4へ線形）
        progress = epoch / args.epochs
        p_aug = 0.7 if progress < 0.8 else (0.7 - (progress - 0.8)/0.2 * (0.7 - 0.4))
        
        # Cutout確率も末期で下げる（期中0.20→期末0.08へ線形）、かつ90%以降は停止
        p_cutout = 0.20 if progress < 0.8 else (0.20 - (progress - 0.8)/0.2 * (0.20 - 0.08))
        if progress >= 0.9:
            p_cutout = 0.0

        # Label smoothing を期末で0.0へ線形減衰（安定収束用）
        if args.label_smoothing_eps > 0.0:
            ls_epoch = args.label_smoothing_eps if progress < 0.8 else max(0.0, args.label_smoothing_eps * (1.0 - (progress - 0.8)/0.2))
            model.label_smoothing_eps = float(ls_epoch)

        # EMA末期強化（90%以降のみEMA decayを0.9997へ一時的に引き上げ）
        ema_decay_backup = model.ema_decay
        if progress >= 0.9:
            model.ema_decay = max(model.ema_decay, 0.9997)
        
        # Training loop
        train_losses = []
        grad_norms = []
        step_sizes = []
        weight_decay_mags = []
        
        for b, (xb, yb) in enumerate(iterate_minibatches(x_tr, y_tr, args.batch_size, shuffle=True, seed=args.seed + epoch)):
            # update LR by scheduler
            model.learning_rate = float(sched.lr_at(step))
            step += 1

            # on-the-fly shift + cutout augmentation (adaptive p): unstandardize -> shift -> cutout -> re-standardize
            xb_raw = xb * std + mean
            applied_aug = False
            if rng_aug_epoch.random() < p_aug:
                # Batch-specific RNG for shift parameters
                rng_shift = np.random.default_rng(args.seed * 1009 + epoch * 131 + b)
                xb_raw = shift_batch_flat(xb_raw, img_hw=(28, 28), max_shift=2, rng=rng_shift)
                applied_aug = True
                aug_applied_count += 1
            
            # Apply Cutout with adaptive probability
            if rng_aug_epoch.random() < p_cutout:
                rng_cut = np.random.default_rng(args.seed * 2029 + epoch * 377 + b)
                xb_raw = cutout_batch_flat(xb_raw, img_hw=(28, 28), max_size=4, rng=rng_cut)
                applied_aug = True
                aug_applied_count += 1

            # Apply small Gaussian noise with low probability (deterministic; gated by flag)
            if args.enable_noise_aug:
                p_noise = 0.2 if progress < 0.8 else 0.1
                if rng_aug_epoch.random() < p_noise:
                    rng_noise = np.random.default_rng(args.seed * 3049 + epoch * 521 + b)
                    xb_raw = gaussian_noise_batch_flat(xb_raw, sigma=0.02, rng=rng_noise)
                    applied_aug = True
                    aug_applied_count += 1

            xb = (xb_raw - mean) / std

            loss, dW, db = model.compute_loss_and_gradients(xb, yb, dropout_rate=current_dropout)
            grad_norm, step_size, wd_mag = model.apply_gradients(dW, db)
            
            train_losses.append(loss)
            grad_norms.append(grad_norm)
            step_sizes.append(step_size)
            weight_decay_mags.append(wd_mag)

        # Validation with MC-Dropout (統一評価, セレクタと同じ30本で決定論的に)
        # use training-time current_dropout for stability of validation metric
        val_probs = model.predict_proba_mc(
            x_val,
            n_samples=30,
            temperature=args.mc_temp,
            seed=args.seed,
            dropout_rate_override=current_dropout,
        )
        val_pred = np.argmax(val_probs, axis=1)
        val_acc = MLPClassifier.accuracy(val_pred, y_val)

        # Compute validation loss for logging (MC-Dropout平均確率でCE)
        val_loss = -np.mean(np.log(val_probs[np.arange(len(y_val)), y_val] + 1e-12))
        avg_val_loss = float(val_loss)

        # Comprehensive logging metrics
        avg_train_loss = float(np.mean(train_losses)) if len(train_losses) > 0 else float('nan')
        avg_grad_norm = float(np.mean(grad_norms)) if len(grad_norms) > 0 else float('nan')
        avg_step_size = float(np.mean(step_sizes)) if len(step_sizes) > 0 else float('nan')
        avg_wd_mag = float(np.mean(weight_decay_mags)) if len(weight_decay_mags) > 0 else float('nan')

        per_class_metrics_epoch = MLPClassifier.per_class_metrics(val_pred, y_val, num_classes)
        per_class_f1 = per_class_metrics_epoch["f1"]

        # Restore EMA decay if it was temporarily lifted
        model.ema_decay = ema_decay_backup

        # A. Learning Health Metrics
        current_lr = model.learning_rate
        nan_inf_check = model.check_nan_inf()
        patience_remaining = args.patience - epochs_no_improve
        
        # B. Regularization Metrics
        activation_stats = model.get_activation_stats(x_val[:1000])  # Sample for efficiency
        weight_norms = model.get_weight_norms()
        
        # C. Data & Augmentation Metrics
        aug_rate = aug_applied_count / (epoch * num_batches)
        std_stats = {"mean_min": float(np.min(mean)), "mean_max": float(np.max(mean)), 
                     "std_min": float(np.min(std)), "std_max": float(np.max(std))}
        
        # D. Error Analysis (every 5 epochs or final epoch)
        if epoch % 5 == 0 or epoch == args.epochs:
            cm = MLPClassifier.confusion_matrix(val_pred, y_val, num_classes)
            calib_metrics = MLPClassifier.calibration_metrics(val_probs, y_val)

            # Top-5 errors (using MC-Dropout probabilities for loss calculation)
            val_losses_flat = -np.log(val_probs[np.arange(len(y_val)), y_val] + 1e-12)
            top_errors = MLPClassifier.top_k_errors(val_pred, y_val, val_losses_flat, k=5)
        else:
            cm = None
            calib_metrics = None
            top_errors = None

        # Print comprehensive metrics
        print(f"Epoch {epoch:03d} | train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f} | val_acc={val_acc:.4f}")
        print(f"  LR={current_lr:.2e} | grad_norm={avg_grad_norm:.3f} | step_size={avg_step_size:.3f} | wd_mag={avg_wd_mag:.3f}")
        print(f"  EarlyStop: best={best_val_acc:.4f}@{best_epoch} | patience={patience_remaining}")
        print(f"  ReLU zeros: {[f'{z:.3f}' for z in activation_stats['zero_rates']]}")
        print(f"  Weight norms: {[f'{w:.3f}' for w in weight_norms['weight_norms']]}")
        print(f"  Aug rate={aug_rate:.3f}")
        
        if nan_inf_check['weights'] or nan_inf_check['biases']:
            print(f"  WARNING: NaN/Inf detected in weights/biases!")
        
        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"  Per-class F1: {[f'{f:.3f}' for f in per_class_f1]}")
            if calib_metrics is not None:
                print(f"  Calibration: ECE={calib_metrics['ece']:.3f} | entropy={calib_metrics['mean_entropy']:.3f} | margin={calib_metrics['mean_margin']:.3f}")
            if top_errors:
                preview = [(e["true"], e["pred"], f"{e['loss']:.3f}") for e in top_errors[:3]]
                print(f"  Top-5 errors: {preview}")

        # SWA snapshot (EMA) late in training
        if args.enable_swa:
            if epoch >= int(np.ceil(args.swa_start_frac * args.epochs)) and (epoch % max(1, args.swa_interval) == 0):
                # snapshot current EMA weights/biases
                swa_ema_snapshots_w.append([w.copy() for w in model.ema_w])
                swa_ema_snapshots_b.append([b.copy() for b in model.ema_b])

        # Early stopping
        if val_acc > best_val_acc + 1e-5:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = model.get_state()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print("Early stopping triggered.")
                break

    if best_state is not None:
        model.set_state(best_state)

    # If SWA enabled and we have snapshots, average EMA params and set them
    if args.enable_swa and len(swa_ema_snapshots_w) > 0:
        num_snaps = len(swa_ema_snapshots_w)
        avg_ema_w = [np.zeros_like(model.ema_w[i]) for i in range(len(model.ema_w))]
        avg_ema_b = [np.zeros_like(model.ema_b[i]) for i in range(len(model.ema_b))]
        for snap_w, snap_b in zip(swa_ema_snapshots_w, swa_ema_snapshots_b):
            for i in range(len(avg_ema_w)):
                avg_ema_w[i] += snap_w[i]
                avg_ema_b[i] += snap_b[i]
        for i in range(len(avg_ema_w)):
            avg_ema_w[i] /= float(num_snaps)
            avg_ema_b[i] /= float(num_snaps)
        model.ema_w = [w.copy() for w in avg_ema_w]
        model.ema_b = [b.copy() for b in avg_ema_b]

    # Retrain on full training data with best hyperparams state? For simplicity, use best_state and fit a few epochs on all data.
    # Standardize full train and test using train statistics
    x_train_std = (x_train - mean) / std
    # Brief fine-tuning on all training data (low LR, light/no aug, low dropout, LS=0)
    ft_epochs = int(args.ft_epochs)
    if ft_epochs > 0:
        base_lr_backup = model.learning_rate
        model.learning_rate = float(max(1e-5, args.lr * 0.2))
        label_smoothing_backup = model.label_smoothing_eps
        model.label_smoothing_eps = 0.0
        dropout_backup = model.dropout_rate
        model.dropout_rate = max(0.0, args.dropout * 0.33)
        try:
            for e in range(ft_epochs):
                for b, (xb, yb) in enumerate(iterate_minibatches(x_train_std, y_train, args.batch_size, shuffle=True, seed=args.seed + 2000 + e)):
                    loss, dW, db = model.compute_loss_and_gradients(xb, yb, dropout_rate=model.dropout_rate)
                    model.apply_gradients(dW, db)
        finally:
            model.learning_rate = base_lr_backup
            model.label_smoothing_eps = label_smoothing_backup
            model.dropout_rate = dropout_backup


    # TTA with optimized EMA swap (1回だけ)
    def tta_predict_proba_5dir_prob(model: MLPClassifier, x_flat: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """5方向確率平均TTA"""
        shifts = [(0,0),(1,0),(-1,0),(0,1),(0,-1)]
        H = W = 28
        N, D = x_flat.shape
        C = model.layer_sizes[-1]
        acc = np.zeros((N, C), dtype=np.float32)
        imgs = x_flat.reshape(N, H, W)
        
        # EMA swap once for all predictions
        if model.use_ema:
            model.swap_to_ema()
            try:
                for dx, dy in shifts:
                    rolled = np.roll(np.roll(imgs, dy, axis=1), dx, axis=2)
                    if dy > 0:   rolled[:, :dy, :] = 0
                    elif dy < 0: rolled[:, dy:, :] = 0
                    if dx > 0:   rolled[:, :, :dx] = 0
                    elif dx < 0: rolled[:, :, dx:] = 0
                    
                    flat = (rolled.reshape(N, D) - mean) / std
                    logits, _ = model.forward(flat, training=False)
                    probs = _softmax_stable(logits)
                    acc += probs
            finally:
                model.swap_from_ema()
        else:
            for dx, dy in shifts:
                rolled = np.roll(np.roll(imgs, dy, axis=1), dx, axis=2)
                if dy > 0:   rolled[:, :dy, :] = 0
                elif dy < 0: rolled[:, dy:, :] = 0
                if dx > 0:   rolled[:, :, :dx] = 0
                elif dx < 0: rolled[:, :, dx:] = 0
                
                flat = (rolled.reshape(N, D) - mean) / std
                logits, _ = model.forward(flat, training=False)
                probs = _softmax_stable(logits)
                acc += probs
        
        return acc / len(shifts)

    def tta_predict_proba_9dir_logits(model: MLPClassifier, x_flat: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """9方向ロジット平均TTA"""
        shifts = [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
        H = W = 28
        N, D = x_flat.shape
        C = model.layer_sizes[-1]
        acc_logits = np.zeros((N, C), dtype=np.float32)
        imgs = x_flat.reshape(N, H, W)
        
        # EMA swap once for all predictions
        if model.use_ema:
            model.swap_to_ema()
            try:
                for dx, dy in shifts:
                    rolled = np.roll(np.roll(imgs, dy, axis=1), dx, axis=2)
                    if dy > 0:   rolled[:, :dy, :] = 0
                    elif dy < 0: rolled[:, dy:, :] = 0
                    if dx > 0:   rolled[:, :, :dx] = 0
                    elif dx < 0: rolled[:, :, dx:] = 0
                    
                    flat = (rolled.reshape(N, D) - mean) / std
                    logits, _ = model.forward(flat, training=False)
                    acc_logits += logits
            finally:
                model.swap_from_ema()
        else:
            for dx, dy in shifts:
                rolled = np.roll(np.roll(imgs, dy, axis=1), dx, axis=2)
                if dy > 0:   rolled[:, :dy, :] = 0
                elif dy < 0: rolled[:, dy:, :] = 0
                if dx > 0:   rolled[:, :, :dx] = 0
                elif dx < 0: rolled[:, :, dx:] = 0
                
                flat = (rolled.reshape(N, D) - mean) / std
                logits, _ = model.forward(flat, training=False)
                acc_logits += logits
        
        # Logit averaging with numerical stability
        logits = acc_logits / len(shifts)
        return _softmax_stable(logits)

    def tta_predict_proba_5dir_center_weighted(model: MLPClassifier, x_flat: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """5方向センター重みTTA（中心に重み2.0、他に重み1.0）"""
        shifts = [(0,0),(1,0),(-1,0),(0,1),(0,-1)]
        weights = [2.0, 1.0, 1.0, 1.0, 1.0]  # 中心に重み2.0
        H = W = 28
        N, D = x_flat.shape
        C = model.layer_sizes[-1]
        acc = np.zeros((N, C), dtype=np.float32)
        imgs = x_flat.reshape(N, H, W)
        
        # EMA swap once for all predictions
        if model.use_ema:
            model.swap_to_ema()
            try:
                for (dx, dy), weight in zip(shifts, weights):
                    rolled = np.roll(np.roll(imgs, dy, axis=1), dx, axis=2)
                    if dy > 0:   rolled[:, :dy, :] = 0
                    elif dy < 0: rolled[:, dy:, :] = 0
                    if dx > 0:   rolled[:, :, :dx] = 0
                    elif dx < 0: rolled[:, :, dx:] = 0
                    
                    flat = (rolled.reshape(N, D) - mean) / std
                    logits, _ = model.forward(flat, training=False)
                    probs = _softmax_stable(logits)
                    acc += probs * weight
            finally:
                model.swap_from_ema()
        else:
            for (dx, dy), weight in zip(shifts, weights):
                rolled = np.roll(np.roll(imgs, dy, axis=1), dx, axis=2)
                if dy > 0:   rolled[:, :dy, :] = 0
                elif dy < 0: rolled[:, dy:, :] = 0
                if dx > 0:   rolled[:, :, :dx] = 0
                elif dx < 0: rolled[:, :, dx:] = 0
                
                flat = (rolled.reshape(N, D) - mean) / std
                logits, _ = model.forward(flat, training=False)
                probs = _softmax_stable(logits)
                acc += probs * weight
        
        # Normalize by total weight
        total_weight = sum(weights)
        return acc / total_weight

    

    

    # 標準化チェック（学習直後かつ標準化データでのみ通る）
    assert abs(float(x_tr.mean())) < 1e-2 and abs(float(x_tr.std() - 1)) < 1e-1, "train not standardized?"
    assert abs(float(x_val.mean())) < 1e-2 and abs(float(x_val.std() - 1)) < 1e-1, "val not standardized?"
    
    # ---- Inference selector on validation ----
    print("Selecting inference mode by validation...")

    # raw arrays for TTA functions
    x_val_raw = x_val * std + mean
    x_test_raw = x_test_std * std + mean

    cands: Dict[str, Tuple[float, object]] = {}

    # 1) Deterministic (EMA)
    p_det = model.predict_proba(x_val)
    acc_det = MLPClassifier.accuracy(np.argmax(p_det, axis=1), y_val)
    cands["det"] = (acc_det, lambda: model.predict_proba(x_test_std))

    # 2) MC-Dropout with expanded temperature grid (optionally overriding dropout during MC)
    best_mc_acc, best_t, best_dr = -1.0, 1.0, None
    best_mc_probs = None
    grid_temps = [0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
    grid_dropouts = [None, 0.15, 0.12, 0.10, 0.075]
    for t in grid_temps:
        for dr in grid_dropouts:
            dr_use = args.mc_dropout_override if args.mc_dropout_override is not None else dr
            p_mc_val = model.predict_proba_mc(
                x_val,
                n_samples=args.mc_samples_test,
                temperature=t,
                seed=args.seed,
                dropout_rate_override=dr_use,
            )
            acc = MLPClassifier.accuracy(np.argmax(p_mc_val, axis=1), y_val)
            if acc > best_mc_acc:
                best_mc_acc, best_t, best_dr = acc, t, dr_use
                best_mc_probs = p_mc_val
    cands[f"mc@{best_t:.2f}"] = (
        best_mc_acc,
        lambda: model.predict_proba_mc(
            x_test_std,
            n_samples=args.mc_samples_test,
            temperature=best_t,
            seed=args.seed,
            dropout_rate_override=best_dr,
        ),
    )

    # 3) MC×TTA (5-direction, logit averaging, total 30 samples) - only MC×TTA, no pure TTA
    p_tta5mc = tta_predict_proba_5dir_logits_mc(
        model, x_val_raw, mean, std,
        temperature=best_t, n_samples_total=args.mc_samples_test, seed=args.seed, dropout_rate_override=best_dr
    )
    acc_tta5mc = MLPClassifier.accuracy(np.argmax(p_tta5mc, axis=1), y_val)
    cands[f"tta5mc@{best_t:.2f}"] = (
        acc_tta5mc,
        lambda: tta_predict_proba_5dir_logits_mc(
            model, x_test_raw, mean, std,
            temperature=best_t, n_samples_total=args.mc_samples_test, seed=args.seed, dropout_rate_override=best_dr
        ),
    )

    # 4) MC×TTA (9-direction) candidate
    p_tta9mc = tta_predict_proba_9dir_logits_mc(
        model, x_val_raw, mean, std,
        temperature=best_t, n_samples_total=args.mc_samples_test, seed=args.seed, dropout_rate_override=best_dr
    )
    acc_tta9mc = MLPClassifier.accuracy(np.argmax(p_tta9mc, axis=1), y_val)
    # 5) Center-weighted 5dir TTA (prob averaging) as candidate (no MC) for diversity
    p_tta5_center = tta_predict_proba_5dir_center_weighted(model, x_val_raw, mean, std)
    acc_tta5_center = MLPClassifier.accuracy(np.argmax(p_tta5_center, axis=1), y_val)
    cands["tta5_center"] = (
        acc_tta5_center,
        lambda: tta_predict_proba_5dir_center_weighted(model, x_test_raw, mean, std)
    )
    cands[f"tta9mc@{best_t:.2f}"] = (
        acc_tta9mc,
        lambda: tta_predict_proba_9dir_logits_mc(
            model, x_test_raw, mean, std,
            temperature=best_t, n_samples_total=args.mc_samples_test, seed=args.seed, dropout_rate_override=best_dr
        ),
    )

    if best_mc_probs is None:
        best_mc_probs = model.predict_proba_mc(
            x_val,
            n_samples=args.mc_samples_test,
            temperature=best_t,
            seed=args.seed,
            dropout_rate_override=best_dr,
        )

    val_cache: Dict[str, np.ndarray] = {
        "det": p_det,
        f"mc@{best_t:.2f}": best_mc_probs,
        f"tta5mc@{best_t:.2f}": p_tta5mc,
        f"tta9mc@{best_t:.2f}": p_tta9mc,
        "tta5_center": p_tta5_center,
    }

    def get_val_probs(name: str) -> np.ndarray:
        if name in val_cache:
            return val_cache[name]
        if name.startswith("mc@"):
            t = float(name.split("@")[1])
            arr = model.predict_proba_mc(
                x_val,
                n_samples=args.mc_samples_test,
                temperature=t,
                seed=args.seed,
                dropout_rate_override=best_dr,
            )
            val_cache[name] = arr
            return arr
        if name.startswith("tta5mc@"):
            t = float(name.split("@")[1])
            arr = tta_predict_proba_5dir_logits_mc(
                model, x_val_raw, mean, std,
                temperature=t,
                n_samples_total=args.mc_samples_test,
                seed=args.seed,
                dropout_rate_override=best_dr,
            )
            val_cache[name] = arr
            return arr
        if name.startswith("tta9mc@"):
            t = float(name.split("@")[1])
            arr = tta_predict_proba_9dir_logits_mc(
                model, x_val_raw, mean, std,
                temperature=t,
                n_samples_total=args.mc_samples_test,
                seed=args.seed,
                dropout_rate_override=best_dr,
            )
            val_cache[name] = arr
            return arr
        # Fallback to deterministic probabilities
        return val_cache.get("det", p_det)

    # Optional: explore blend between top-2 candidates even when not tied
    sorted_cands = sorted(cands.items(), key=lambda kv: kv[1][0], reverse=True)
    alpha_grid = [i / 10.0 for i in range(2, 9)]  # 0.2 .. 0.8

    if len(sorted_cands) >= 2:
        (name1, (acc1, fn1)), (name2, (acc2, fn2)) = sorted_cands[:2]
        p1 = get_val_probs(name1)
        p2 = get_val_probs(name2)
        best_alpha = None
        best_blend_acc = -1.0
        best_blend_probs = None
        for alpha in alpha_grid:
            mix = alpha * p1 + (1.0 - alpha) * p2
            acc_mix = MLPClassifier.accuracy(np.argmax(mix, axis=1), y_val)
            if acc_mix > best_blend_acc:
                best_blend_acc = acc_mix
                best_alpha = alpha
                best_blend_probs = mix
        if best_alpha is not None and best_blend_acc > max(acc1, acc2) + 1e-4:
            blend_name = f"blend2:{best_alpha:.2f}*{name1}+(1-{best_alpha:.2f})*{name2}"
            val_cache[blend_name] = best_blend_probs
            cands[blend_name] = (
                best_blend_acc,
                lambda fn_primary=fn1, fn_secondary=fn2, alpha=best_alpha: (
                    alpha * fn_primary() + (1.0 - alpha) * fn_secondary()
                ),
            )

    # Three-way blending disabled (validation gains did not translate to LB)

    # pick best by validation with tie-breaking blend logic
    eps = 3e-4  # tolerance for "equal" performance (tighter for blending)
    
    # Find the best accuracy with tolerance
    best_acc = max(acc for acc, _ in cands.values())
    cands_tie = [(n, a, f) for n, (a, f) in cands.items() if abs(a - best_acc) <= eps]
    
    if len(cands_tie) >= 3:
        # 上位3つの平均ブレンド
        cands_tie.sort(key=lambda x: x[1], reverse=True)
        (name1, acc1, fn1), (name2, acc2, fn2), (name3, acc3, fn3) = cands_tie[:3]
        best_name = f"blend3:{name1}+{name2}+{name3}"
        best_acc = (acc1 + acc2 + acc3) / 3.0
        def best_fn():
            return (fn1() + fn2() + fn3()) / 3.0
    elif len(cands_tie) == 2:
        if args.enable_adaptive_blend:
            # 上位2つの適応的ブレンド（alphaを探索してval精度最大の重みを選択）
            cands_tie.sort(key=lambda x: x[1], reverse=True)
            (name1, acc1, fn1), (name2, acc2, fn2) = cands_tie[:2]
            p1 = get_val_probs(name1)
            p2 = get_val_probs(name2)
            best_local_acc = -1.0
            best_alpha = 0.5
            best_mix = None
            for alpha in alpha_grid:
                pv = alpha * p1 + (1.0 - alpha) * p2
                acc_v = MLPClassifier.accuracy(np.argmax(pv, axis=1), y_val)
                if acc_v > best_local_acc:
                    best_local_acc = acc_v
                    best_alpha = alpha
                    best_mix = pv
            best_name = f"blend:{best_alpha:.2f}*{name1}+(1-{best_alpha:.2f})*{name2}"
            best_acc = best_local_acc
            if best_mix is not None:
                val_cache[best_name] = best_mix
            def best_fn():
                return best_alpha * fn1() + (1.0 - best_alpha) * fn2()
        else:
            # 固定0.5ブレンド
            cands_tie.sort(key=lambda x: x[1], reverse=True)
            (name1, acc1, fn1), (name2, acc2, fn2) = cands_tie[:2]
            best_name = f"blend:{name1}+{name2}"
            best_acc = (acc1 + acc2) / 2.0
            def best_fn():
                return 0.5 * fn1() + 0.5 * fn2()
    else:
        # 単一候補または差が大きい場合は従来の優先順位ロジック
        priority_order = ["det", "mc@0.85", "mc@0.90", "mc@0.95", "mc@1.00", "tta5mc@0.85", "tta5mc@0.90", "tta5mc@0.95", "tta5mc@1.00"]
        
        best_name, best_acc, best_fn = None, -1.0, None
        for priority_name in priority_order:
            for name, acc, fn in cands_tie:
                if (name == priority_name or 
                    (name.startswith("mc@") and priority_name.startswith("mc@")) or
                    (name.startswith("tta5mc@") and priority_name.startswith("tta5mc@"))):
                    best_name, best_acc, best_fn = name, acc, fn
                    break
            if best_name is not None:
                break
        
        # Fallback to highest accuracy if no priority match
        if best_name is None:
            best_name, (best_acc, best_fn) = max(cands.items(), key=lambda kv: kv[1][0])
    
    print(
        f"[Selector] best={best_name} val_acc={best_acc:.4f} "
        f"(det={acc_det:.4f}, mc@{best_t:.2f}={best_mc_acc:.4f}, tta5mc@{best_t:.2f}={acc_tta5mc:.4f}, tta9mc@{best_t:.2f}={acc_tta9mc:.4f})"
    )

    # Uncertainty gate path removed in revert-to-baseline step
    gate_used = False
    gate_info = None

    # Optional: override inference choice
    if args.force_infer is not None:
        forced = args.force_infer.strip()
        def parse_T(s: str) -> float:
            try:
                return float(s.split("@")[1])
            except Exception:
                return best_t
        if forced == 'det':
            best_name = 'det'
            def best_fn():
                return model.predict_proba(x_test_std)
        elif forced.startswith('mc@'):
            T = parse_T(forced)
            best_name = f'mc@{T:.2f}'
            def best_fn():
                return model.predict_proba_mc(x_test_std, n_samples=args.mc_samples_test, temperature=T, seed=args.seed, dropout_rate_override=best_dr)
        elif forced.startswith('tta5mc@'):
            T = parse_T(forced)
            best_name = f'tta5mc@{T:.2f}'
            def best_fn():
                return tta_predict_proba_5dir_logits_mc(model, x_test_raw, mean, std, temperature=T, n_samples_total=args.mc_samples_test, seed=args.seed, dropout_rate_override=best_dr)
        elif forced.startswith('tta9mc@'):
            T = parse_T(forced)
            best_name = f'tta9mc@{T:.2f}'
            def best_fn():
                return tta_predict_proba_9dir_logits_mc(model, x_test_raw, mean, std, temperature=T, n_samples_total=args.mc_samples_test, seed=args.seed, dropout_rate_override=best_dr)
        print(f"[ForceInfer] forcing inference mode: {best_name}")

    # final prediction (optional post-hoc temperature scaling)
    proba = best_fn()
    if args.enable_temp_scale and (not gate_used):
        # calibrate one scalar T on validation for the chosen candidate name
        # recompute chosen candidate's validation probs deterministically
        def get_val_probs_for_final(name: str) -> np.ndarray:
            return get_val_probs(name)

        p_val_sel = get_val_probs_for_final(best_name)
        # search T in a small grid around 1.0
        Ts = [0.85, 0.9, 0.95, 1.0, 1.05, 1.1]
        best_T = 1.0
        best_T_acc = MLPClassifier.accuracy(np.argmax(p_val_sel, axis=1), y_val)
        logits_val = np.log(np.clip(p_val_sel, 1e-12, 1.0))  # up to a const shift
        for T in Ts:
            scaled = logits_val / max(T, 1e-6)
            scaled = scaled - scaled.max(axis=1, keepdims=True)
            pv = np.exp(scaled)
            pv /= pv.sum(axis=1, keepdims=True)
            accT = MLPClassifier.accuracy(np.argmax(pv, axis=1), y_val)
            if accT > best_T_acc:
                best_T_acc = accT
                best_T = T
        if best_T != 1.0:
            # apply to test proba
            logits_test = np.log(np.clip(proba, 1e-12, 1.0))
            logits_test = logits_test / max(best_T, 1e-6)
            logits_test = logits_test - logits_test.max(axis=1, keepdims=True)
            proba = np.exp(logits_test)
            proba /= proba.sum(axis=1, keepdims=True)
    y_pred = np.argmax(proba, axis=1)

    # Optionally save test probabilities for ensembling
    if args.save_test_proba:
        proba_out = args.proba_path
        if not proba_out:
            proba_out = os.path.join(args.output_dir, f"proba_seed{args.seed}.npy")
        np.save(proba_out, proba.astype(np.float32))
        # sidecar meta for weighted ensemble
        side_meta = {
            "selector_best_name": best_name,
            "selector_val_acc": float(best_acc),
            "mc_best_temp": float(best_t),
            "mc_best_dropout_override": None if best_dr is None else float(best_dr),
            "seed": int(args.seed),
        }
        with open(proba_out + ".meta.json", "w", encoding="utf-8") as f:
            json.dump(side_meta, f)
        print(f"Saved test probabilities to {proba_out} and meta to {proba_out}.meta.json")

    # Save artifacts
    pred_path = os.path.join(args.output_dir, "y_pred.csv")
    np.savetxt(
        pred_path,
        y_pred.astype(np.int64),
        fmt="%d",
        delimiter=",",
        newline="\n",
        header="label",
        comments="",
    )

    model_path = os.path.join(args.model_dir, "mlp_model.npz")
    meta: Dict[str, any] = {
        "best_val_acc": float(best_val_acc),
        "best_inference_method": best_name,
    }
    model.save_npz(model_path, standardization_mean=mean, standardization_std=std, meta=meta)

    print(f"Saved predictions to {pred_path}")
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
