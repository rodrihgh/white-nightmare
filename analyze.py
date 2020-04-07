import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import main
from tools import signal_processing as sp
from tools.media import frameshow


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

    print("Generating Data.")
    root, children = sp.wilson_algorithm(main.maze_height, main.maze_width)
    maze, maze_history = sp.render_maze(root, children, main.alpha_maze, **main.bg_args)

    halfway = maze_history.shape[0] // 2
    maze_mask = sp.resample(maze_history[halfway,], order=0, factors=main.wall_thickness)

    maze = sp.resample(maze, factors=main.wall_thickness)
    maze = np.repeat(maze[None,], avg_frames, axis=0)

    background = sp.background((avg_frames, main.maze_height, main.maze_width),
                               **main.bg_args, upsampling=(1,) + (main.wall_thickness,) * 2)

    maze, background = (sp.add_wgn(data, main.noise_std) for data in (maze, background))
    noise_dev = background.std()

    print(f"Empiric standard deviation of noise background: {noise_dev}.")
    print("This value will be use to generate pure white noise")

    noise = sp.normal(scale=noise_dev, size=background.shape)

    maze_bg, maze_noise = (np.where(maze_mask, maze[mix_frame,], data[mix_frame,])[None,]
                           for data in (background, noise))

    maze, background, maze_bg, maze_noise = (sp.shift_video(data, main.alpha_shift, main.alpha_t, main.max_shift)
                                             for data in (maze, background, maze_bg, maze_noise))

    frame_plot = tuple(np.squeeze(data) for data in (maze_bg, maze_noise))
    v_lim = (min(d.min() for d in frame_plot), max(d.max() for d in frame_plot))

    for fp in frame_plot:
        frameshow(fp, v=v_lim)

    print("Plotting histograms.")
    specs = []
    for data, name in zip((maze, background, noise), names):
        mu = data.mean()
        sigma = data.std()

        _, ax = plt.subplots()
        n, bins, patches = ax.hist(data.reshape(-1), bins=hist_bins, density=True)

        # add a 'best fit' line
        pdf = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
               np.exp(-0.5 * (1 / sigma * (bins - mu)) ** 2))
        ax.plot(bins, pdf, '--')
        plt.xlabel("Pixel brightness")
        plt.ylabel("Probability density")
        plt.xlim(-3, 3)
        plt.ylim(0, .75)
        plt.title(rf"Histogram for {name}. $\mu={np.round(mu, 2)}$, $\sigma={np.round(sigma, 2)}$")
        spec = spectrum(data, nfft=data.shape[-1] * zero_pad)
        del data

        specs.append(spec)

    ref_db = max(s.max() for s in specs)
    specs = tuple(s - ref_db for s in specs)
    c_lim = (min(s.min() for s in specs), 0)

    fmax = .5

    print("Plotting spectra.")
    for spec, name in zip(specs, names):
        n_samples = spec.shape[0]
        f = np.linspace(-fmax, fmax, n_samples)
        spec_slice = spec[:, n_samples // 2]

        plt.figure()
        plt.imshow(spec, vmin=c_lim[0], vmax=c_lim[1], extent=(-fmax, fmax, -fmax, fmax))
        cbar = plt.colorbar()
        cbar.set_label("Power level [dB]")
        plt.xlabel(r"Horizontal spatial frequency $f_h$ [cycles/pixel]")
        plt.ylabel(r"Vertical spatial frequency $f_v$ [cycles/pixel]")
        plt.title(f"2D-spectrum for {name}")

        _, ax = plt.subplots()
        multicolor_line(f, spec_slice, ax, v=c_lim)
        plt.xlabel("Horizontal spatial frequency [cycles/pixel]")
        plt.ylabel("Power level [dB]")
        plt.title(rf"Horizontal spectrum slice for {name} @$f_v=0$")

    plt.show()
