import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import main
from tools import signal_processing as sp
from tools.media import frameshow, media_dir


def spectrum(array, nfft, norm=False):
    """
    Calculate the averaged log-spectrum.
    :param array: Input array.
    :param nfft: Number of FFT points.
    :param norm: If True, spectrum is normalized to full scale.
    :return: Calculated spectrum.
    """
    complex_spec = np.fft.fft2(array, s=(nfft, nfft), axes=(-2, -1))
    real_spec = np.abs(complex_spec)
    db_spec = sp.mag2db(real_spec)
    avg_spec = np.mean(db_spec, axis=0)
    centered_spec = np.fft.fftshift(avg_spec)

    if norm:
        centered_spec -= centered_spec.max()

    return centered_spec


def multicolor_line(x, y, axes, c=None, v=None, cmap='viridis'):
    """
    Plot a multicolored line according to some scale
    :param x: x-axis values
    :param y: y-axis values
    :param axes: the axes object to plot onto
    :param c: Scale. if not given, y is taken
    :param v: maximum and minimum color values. If not given, take the scale's max and min.
    :param cmap: colourmap
    :return: the Line object
    """
    if c is None:
        c = y
    if v is None:
        v = (y.min(), y.max())

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(*v)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    # Set the values used for colormapping
    lc.set_array(c)
    lc.set_linewidth(2)
    line = axes.add_collection(lc)
    axes.set_xlim(x.min(), x.max())
    axes.set_ylim(*v)

    return line


if __name__ == "__main__":

    names = ("Maze", "Background noise", "White noise")
    hist_bins = 100
    avg_frames = 64
    zero_pad = 4
    mix_frame = 0

    grayscale = 'gist_gray'

    print("Generating Data.")
    root, children = sp.wilson_algorithm(main.maze_height, main.maze_width)
    maze, maze_history = sp.render_maze(root, children, main.alpha_maze, **main.bg_args)

    halfway = maze_history.shape[0] // 2
    maze_mask = sp.resample(maze_history[halfway,], order=0, factors=main.wall_thickness)

    maze = sp.resample(maze, factors=main.wall_thickness)
    maze = np.repeat(maze[None,], avg_frames, axis=0)

    background = sp.background((avg_frames, main.maze_height * 2, main.maze_width * 2),
                               **main.bg_args, upsampling=(1,) + (main.wall_thickness,) * 2)

    maze, background = (sp.add_wgn(data, main.noise_std) for data in (maze, background))
    noise_dev = background.std()

    print(f"Empiric standard deviation of noise background: {noise_dev}.")
    print("This value will be use to generate pure white noise")

    noise = sp.normal(scale=noise_dev, size=background.shape)

    maze_frame, bg_frame, noise_frame = (data[mix_frame, ] for data in (maze, background, noise))

    maze_bg, maze_noise = (np.where(maze_mask, maze_frame, data)[None,]
                           for data in (bg_frame, noise_frame))

    maze, background, maze_bg, maze_noise = (sp.shift_video(data, main.alpha_shift, main.alpha_t, main.max_shift)
                                             for data in (maze, background, maze_bg, maze_noise))

    bg_frame = background[mix_frame, ]

    frame_plot = tuple(np.squeeze(data) for data in (bg_frame, noise_frame))
    v_lim = (min(d.min() for d in frame_plot), max(d.max() for d in frame_plot))

    for fp, fname in zip(frame_plot, ("bg-frame.svg", "noise-frame.svg")):
        frameshow(fp, v=v_lim)
        plt.savefig(media_dir / fname, transparent=True)

    fig, axs = plt.subplots(1, 2, frameon=False)

    for data, ax2 in zip((maze_bg, maze_noise), axs):
        ax2.imshow(np.squeeze(data), vmin=v_lim[0], vmax=v_lim[1], cmap=grayscale)
        ax2.axis('off')
    fig.tight_layout()

    plt.savefig(media_dir / "transition.svg")

    print("Plotting histograms.")
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.85), frameon=False)

    specs = []
    for data, name, ax in zip((maze, background, noise), names, axs):
        mu = data.mean()
        sigma = data.std()

        n, bins, patches = ax.hist(data.reshape(-1), bins=hist_bins, density=True)

        # add a 'best fit' line
        pdf = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
               np.exp(-0.5 * (1 / sigma * (bins - mu)) ** 2))
        ax.plot(bins, pdf, '--')
        ax.set_xlabel("Pixel brightness")
        if name == names[0]:
            ax.set_ylabel("Probability density")
        ax.set_xlim(-3, 3)
        ax.set_ylim(0, .75)
        ax.set_title(f"{name}.\n" + rf"$\mu={np.round(mu, 2)}$, $\sigma={np.round(sigma, 2)}$")
        spec = spectrum(data, nfft=data.shape[-1] * zero_pad)
        del data

        specs.append(spec)

    plt.savefig(media_dir / "histograms.svg")

    ref_db = max(s.max() for s in specs)
    specs = tuple(s - ref_db for s in specs)
    c_lim = (min(s.min() for s in specs), 0)

    fmax = .5

    print("Plotting spectra.")
    for spec, name in zip(specs, names):
        n_samples = spec.shape[0]
        f = np.linspace(-fmax, fmax, n_samples)
        spec_slice = spec[:, n_samples // 2]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.25, 4), frameon=False)

        fig.suptitle(f"Spectrum of {name}")

        im = ax1.imshow(spec, vmin=c_lim[0], vmax=c_lim[1], extent=(-fmax, fmax, -fmax, fmax))

        ax1.set_xlabel(r"Horizontal spatial frequency $f_h$ [cycles/pixel]")
        ax1.set_ylabel(r"Vertical spatial frequency $f_v$ [cycles/pixel]")
        ax1.set_title("2D-spectrum")

        multicolor_line(f, spec_slice, ax2, v=c_lim)
        ax2.set_xlabel("Horizontal spatial frequency [cycles/pixel]")
        ax2.get_yaxis().set_visible(False)
        ax2.set_title(r"Horizontal spectrum slice @ $f_v=0$")

        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label("Power level [dB]")

        plt.savefig(media_dir / (name + " spectrum.svg"))

    # plt.show()
