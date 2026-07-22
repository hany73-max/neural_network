import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# ---------------------------------------------------------------------------
# From-scratch PCA — used to project high-dimensional layer activations
# (e.g. a 1000-neuron layer's output) down to 2D so they can be plotted.
# Implemented via eigen-decomposition of the covariance matrix, no sklearn.
# ---------------------------------------------------------------------------
def pca_project(A, n_components=2):
    """
    A: (n_samples, n_features) activation matrix from one layer.
    Returns (n_samples, n_components). If A already has <= n_components
    features, it's zero-padded instead of reduced (nothing to project).
    """
    m, n = A.shape
    if n <= n_components:
        pad_width = n_components - n
        return np.hstack([A, np.zeros((m, pad_width))]) if pad_width > 0 else A

    A_centered = A - A.mean(axis=0, keepdims=True)
    cov = np.cov(A_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)          # ascending order
    top = np.argsort(eigvals)[::-1][:n_components]  # largest variance first
    return A_centered @ eigvecs[:, top]


# ---------------------------------------------------------------------------
# 1. Cost over epochs — static line plot
# ---------------------------------------------------------------------------
def plot_cost_curve(costs, title="Training Cost", ax=None, show=True):
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(range(1, len(costs) + 1), costs, color="#d63384", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cost (MSE)")
    ax.set_title(title)
    ax.grid(alpha=0.3)

    if own_fig and show:
        plt.tight_layout()
        plt.show()
    return ax


# ---------------------------------------------------------------------------
# 2. Predicted curve vs. true curve, animated across training checkpoints
# ---------------------------------------------------------------------------
def animate_predictions(X_real, y_real, snapshots, interval=150, save_path=None):
    """
    X_real, y_real : the UNSCALED X and true y (for a human-readable axis).
    snapshots      : list of (epoch, A_final) tuples, as returned by train().
    save_path      : if given, saves an animated GIF instead of showing it live.
    """
    X_flat = X_real.flatten()
    y_flat = y_real.flatten()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(X_flat, y_flat, label="True y", color="black", linewidth=2)
    pred_line, = ax.plot([], [], label="Predicted", color="#e8590c", linewidth=2)
    epoch_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    ax.set_xlim(X_flat.min(), X_flat.max())
    margin = 0.1 * (y_flat.max() - y_flat.min())
    ax.set_ylim(y_flat.min() - margin, y_flat.max() + margin)
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.set_title("Prediction vs. True Curve Over Training")
    ax.legend(loc="upper right")

    def update(frame_idx):
        epoch, A_final = snapshots[frame_idx]
        pred_line.set_data(X_flat, A_final.flatten())
        epoch_text.set_text(f"Epoch {epoch}")
        return pred_line, epoch_text

    anim = animation.FuncAnimation(
        fig, update, frames=len(snapshots), interval=interval, blit=True
    )

    if save_path:
        anim.save(save_path, writer="pillow")
    else:
        plt.show()

    return anim


# ---------------------------------------------------------------------------
# 3. Data-space evolution, layer by layer, animated across checkpoints.
#    Automatically adapts to however many layers the network has —
#    change n_layers in main.py and this needs no edits.
# ---------------------------------------------------------------------------
def animate_layer_space(layer_snapshots, y_real=None, interval=200, save_path=None):
    """
    layer_snapshots : list of (epoch, [A_layer1, A_layer2, ..., A_layerN])
                       as returned by train(). N adapts automatically.
    y_real          : optional true y values, used only to color points
                       (so you can see how the network separates/orders
                       samples by target value as training progresses).
    """
    n_layers = len(layer_snapshots[0][1])
    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 4))
    if n_layers == 1:
        axes = [axes]

    colors = y_real.flatten() if y_real is not None else None
    scatters = []
    for i, ax in enumerate(axes):
        scat = ax.scatter([], [], s=14, c=[], cmap="viridis")
        scatters.append(scat)
        ax.set_title(f"Layer {i + 1} output space")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")

    epoch_text = fig.text(0.02, 0.95, "")

    def update(frame_idx):
        epoch, layer_outputs = layer_snapshots[frame_idx]
        for i, A in enumerate(layer_outputs):
            proj = pca_project(A, n_components=2)
            scatters[i].set_offsets(proj)
            if colors is not None:
                scatters[i].set_array(colors)
            axes[i].relim()
            axes[i].autoscale_view()
        epoch_text.set_text(f"Epoch {epoch}")
        return scatters + [epoch_text]

    anim = animation.FuncAnimation(
        fig, update, frames=len(layer_snapshots), interval=interval, blit=False
    )

    if save_path:
        anim.save(save_path, writer="pillow")
    else:
        plt.show()

    return anim


# ---------------------------------------------------------------------------
# Convenience: run all three for one trained model's results in one call.
# ---------------------------------------------------------------------------
def visualize_training_run(X_real, y_real, A_finals, layer_snapshots, costs,
                            save_prefix=None):
    """
    save_prefix: if given (e.g. "runs/10_neurons"), saves
                 "{save_prefix}_cost.png", "{save_prefix}_predictions.gif",
                 "{save_prefix}_layer_space.gif" instead of showing them live.
    """
    if save_prefix:
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_cost_curve(costs, ax=ax, show=False)
        fig.tight_layout()
        fig.savefig(f"{save_prefix}_cost.png")
        plt.close(fig)
    else:
        plot_cost_curve(costs)

    pred_path = f"{save_prefix}_predictions.gif" if save_prefix else None
    animate_predictions(X_real, y_real, A_finals, save_path=pred_path)

    space_path = f"{save_prefix}_layer_space.gif" if save_prefix else None
    animate_layer_space(layer_snapshots, y_real=y_real, save_path=space_path)