import numpy as np
from numpy import clip
from numpy.random import normal, binomial, choice
from scipy import signal, ndimage
import warnings


def resample(array, factors, order=3, mode='mirror'):
    """
    Resample multidimensional arrays.
    :param array: Input array.
    :param factors: Resampling factor for each axis.
    :param order: Order of spline interpolation.
    :param mode: Extension mode of the input beyond its boundaries
    :return: Resampled array
    """
    return ndimage.zoom(array, factors, order=order, mode=mode)


def ar_filter(array, alpha, axis):
    """
    Filter a multidimensional array according to an AR(1) process.
    :param array: Input array.
    :param alpha: Parameter of AR(1).
    :param axis: axis where filtering is applied.
    :return: Filtered array
    """

    b = [1 - alpha]
    a = [1, -alpha]

    ar = signal.lfilter(b, a, array, axis=axis)

    return ar


def ar_gain(alpha):
    """
    Calculate ratio between the standard deviation of the noise term in an AR(1) process and the resultant
    standard deviation of the AR(1) process.
    :param alpha: Parameter of AR(1)
    :return: Ratio between std of noise term and std of AR(1)
    """
    return np.sqrt((1 + alpha) / (1 - alpha))


def db2mag(db):
    return 10 ** (db / 20)


def db2power(db):
    return 10 ** (db / 10)


def mag2db(mag):
    return 20 * np.log10(mag)


def get_alpha(attenuation, lag):
    """
    Calculate the alpha of AR(1) so that the attenuation condition at the specified lag is fulfilled.
    :param attenuation: Desired attenuation at lag in dB. Must be positive.
    :param lag: Reference lag period in samples. Must be greater or equal than 2.
    :return: Calculated alpha.
    """
    tol = 0.01

    if lag < 2:
        raise ValueError(f"Lag must be greater or equal than 2. Input value is {lag}")
    elif attenuation <= 0:
        raise ValueError(f"Attenuation must be positive. Input value is {attenuation}")

    w = 2 * np.pi / lag
    rho = db2power(attenuation)

    cos_w = np.cos(w)
    b = (rho - cos_w) / (rho - 1)
    alpha = b - np.sqrt(b ** 2 - 1)

    if alpha >= (1 - tol):
        warnings.warn(f"Alpha value equals {alpha}. This value is too close to one and may cause instability",
                      RuntimeWarning)

    return alpha


def background(shape, white, black, std, p=0.5, upsampling=None, order=3):
    """
    Generate noisy background as a sum of binomial and gaussian
    :param shape: shape of the generated background
    :param white: Numeric value of white
    :param black: Numeric value of black
    :param std: Standard deviation of gaussian term
    :param p: Probability of the binomial term
    :param upsampling: Upsampling factors for each axis. Default is None, meaning no upsampling
    :param order: Interpolation order for upsampling, if applied. Defaults to 3.
    :return: Noisy background
    """

    bg = (white - black) * binomial(1, p=p, size=shape) + normal(loc=black, scale=std, size=shape)
    if upsampling is not None:
        bg = resample(bg, upsampling, order=order)
    return bg


def trunc_norm(loc=0.0, scale=1.0, a=-1.0, b=1.0, size=None):
    """
    Generate samples according to a truncated normal distribution.
    :param loc: Mean of the distribution
    :param scale: Standard deviation.
    :param a: Lower bound.
    :param b: Upper bound.
    :param size: Size of the returned samples
    :return: Array of shape size whose values follow a truncated normal distribution.
    """
    x = normal(loc, scale, size)

    if size is None:
        while not a <= x <= b:
            x = normal(loc, scale, size)
    else:
        invalid = np.where(np.logical_or(x < a, b < x))
        invalid_count = len(invalid[0])
        while invalid_count > 0:
            x[invalid] = normal(loc, scale, size=invalid_count)
            invalid = np.where(np.logical_or(x < a, b < x))
            invalid_count = len(invalid[0])

    return x


def wilson_algorithm(maze_height, maze_width, root_cell="SE"):
    """
    Graph tree generation of a maze using Wilson algorithm.

    :param maze_width: Width of the squared maze in cells.
    :param maze_height: Height of the squared maze in cells.
    :param root_cell: Selection of the initial cell.
    :return: Coordinates of root cell and nested list where children cells are indexed by their root's coordinates.
    """
    starting_values = {"NW": (0, 0),
                       "NE": (0, maze_width - 1),
                       "SW": (maze_height - 1, 0),
                       "SE": (maze_height - 1, maze_width - 1),
                       "CENTER": (maze_height // 2, maze_width // 2),
                       "RANDOM": (choice(maze_height), choice(maze_width))}

    if root_cell not in starting_values:
        raise ValueError(f"Unrecognized root cell {root_cell}."
                         f"Permitted values are {tuple(starting_values.keys())}")
    else:
        root = starting_values[root_cell]

    compass = ((-1, 0), (0, 1), (1, 0), (0, -1))

    def get_neighbors(index):
        neighbor_list = []
        for c in compass:
            cell_i = index[0] + c[0]
            cell_j = index[1] + c[1]
            if 0 <= cell_i < maze_height and 0 <= cell_j < maze_width:
                neighbor_list.append((cell_i, cell_j))
        return neighbor_list

    children = [[[] for _ in range(maze_width)] for _ in range(maze_height)]
    maze_cells = np.zeros((maze_height, maze_width), dtype=np.bool)
    maze_cells[root] = True

    while not np.all(maze_cells):
        visit_i, visit_j = np.where(np.logical_not(maze_cells))
        visit_index = choice(len(visit_i))
        cell2visit = (visit_i[visit_index], visit_j[visit_index])
        visited_path = [cell2visit]
        visited_cells = np.zeros_like(maze_cells)
        visited_cells[cell2visit] = True
        while not maze_cells[cell2visit]:
            neighbors = get_neighbors(cell2visit)
            i_neighbor = choice(len(neighbors))
            cell2visit = neighbors[i_neighbor]
            if not visited_cells[cell2visit]:
                visited_path.append(cell2visit)
                visited_cells[cell2visit] = True
            else:
                loop_cell = cell2visit
                cell2visit = visited_path[-1]
                while cell2visit != loop_cell:
                    visited_cells[cell2visit] = False
                    del visited_path[-1]
                    cell2visit = visited_path[-1]
        reverse_path = visited_path[::-1]

        maze_cells = np.logical_or(maze_cells, visited_cells)
        for n in range(len(reverse_path) - 1):
            curr_cell = reverse_path[n + 1]
            prev_cell = reverse_path[n]
            children[prev_cell[0]][prev_cell[1]].append(curr_cell)

    return root, children


def render_maze(root, children, alpha, std, black=0.0, white=1.0, history=True):
    """
    Render maze walls and cells according to an autoregressive model.

    :param root: Coordinates of root cell.
    :param children: Nested list where children cells are indexed by their root's coordinates.
    :param alpha: Parameter of the AR(1) process
    :param std: Standard deviation of the AR(1) process.
    :param black: Value of the black color associated to walls.
    :param white: Value of the white color associated to cells.
    :param history: Boolean indicating whether to keep a frame-by-frame record of the rendering.
    :return: Rendered maze. If history is set, this is also returned as a frame by frame mask.
    """

    sigma = std * ar_gain(alpha)

    def x_ar(*x_past, loc=black):
        x_curr = alpha * np.sum(x_past) + (1 - len(x_past) * alpha) * normal(loc=loc, scale=sigma)
        return x_curr

    edge_height = len(children) * 2
    edge_width = len(children[0]) * 2

    current_branches = [root]

    rendered_maze = np.zeros(shape=(edge_height, edge_width), dtype=np.float)
    maze_mask = np.zeros(shape=rendered_maze.shape, dtype=np.bool) if history else None

    i_root = 2 * root[0]
    j_root = 2 * root[1]

    for i in range(-1, 2):
        for j in range(-1, 2):
            if history:
                maze_mask[i_root + i, j_root + j] = True
            if i == 0 and j == 0:
                rendered_maze[i_root, j_root] = normal(loc=white, scale=sigma)
            else:
                rendered_maze[i_root + i, j_root + j] = normal(loc=black, scale=sigma)

    maze_history = maze_mask[None, ] if history else None

    while len(current_branches) > 0:
        future_branches = []
        for branch in current_branches:
            for child in children[branch[0]][branch[1]]:

                # Calculate position of cells to render
                direction = (child[0] - branch[0], child[1] - branch[1])

                mid_cell = (branch[0] + child[0], branch[1] + child[1])
                end_cell = (2 * child[0], 2 * child[1])

                void1a = (2 * child[0] + direction[1], 2 * child[1] + direction[0])
                void1b = (2 * child[0] - direction[1], 2 * child[1] - direction[0])
                void2a = (2 * child[0] + direction[0] + direction[1], 2 * child[1] + direction[1] + direction[0])
                void2b = (2 * child[0] + direction[0] - direction[1], 2 * child[1] + direction[1] - direction[0])
                end_void = (2 * child[0] + direction[0], 2 * child[1] + direction[1])

                # Fetch value of already rendered cells
                w0 = rendered_maze[2 * branch[0], 2 * branch[1]]

                b0a = rendered_maze[branch[0] + child[0] + direction[1], branch[1] + child[1] + direction[0]]
                b0b = rendered_maze[branch[0] + child[0] - direction[1], branch[1] + child[1] - direction[0]]

                # Calculate value of newly rendered cells
                w1 = x_ar(w0, loc=white)
                w2 = x_ar(w1, loc=white)

                b1a = x_ar(b0a, loc=black)
                b1b = x_ar(b0b, loc=black)
                b2a = x_ar(b1a, loc=black)
                b2b = x_ar(b1b, loc=black)
                b3 = x_ar(b2a, b2b, loc=black)

                # Assign rendered values
                rendered_maze[mid_cell] = w1
                rendered_maze[end_cell] = w2

                rendered_maze[void1a] = b1a
                rendered_maze[void1b] = b1b
                rendered_maze[void2a] = b2a
                rendered_maze[void2b] = b2b
                rendered_maze[end_void] = b3

                if history:

                    maze_mask[mid_cell] = True
                    maze_mask[end_cell] = True

                    maze_mask[void1a] = True
                    maze_mask[void1b] = True
                    maze_mask[void2a] = True
                    maze_mask[void2b] = True
                    maze_mask[end_void] = True

                future_branches += [child]

        current_branches = future_branches
        if history:
            maze_history = np.append(maze_history, maze_mask[None, ], axis=0)

    if history:
        return rendered_maze, maze_history
    else:
        return rendered_maze


def render_transition(maze, maze_history, fadein_frames=0, alpha_stop=0.1,
                      std=0.0, black=0.0, white=1.0, upfactor=None):
    """
    Render the transition from white noise to maze
    :param maze: Final rendered maze
    :param maze_history: Frame-by-frame mask for the transition
    :param std: Standard deviation of the background.
    :param black: Value of the black color associated to walls.
    :param white: Value of the white color associated to cells.
    :param fadein_frames: Number of additional frames for fade in. Default is 0, meaning no fade in.
    :param alpha_stop: Percentage from where fade-in is considered finished.
    :param upfactor: Upsampling factor for the maze. Default is None, meaning no upsampling
    :return: Rendered transition
    """

    maze_frames = maze_history.shape[0]
    canvas_shape = (maze_frames + fadein_frames,) + maze.shape

    maze_mask = maze_history.astype(np.float)

    if fadein_frames < 0:
        raise ValueError("fadein_frames must be a non-negative integer")
    elif fadein_frames > 0:
        alpha_fadein = alpha_stop ** (1 / fadein_frames)

        final_mask = maze_mask[-1:, ]

        maze_mask = insert_frames(maze_mask, fadein_frames, pos=-1, ref=-1)

        maze_mask = ar_filter(maze_mask, alpha_fadein, axis=0)

        maze_mask[-1, ] = final_mask

    if upfactor is not None:
        upsampling = (1, upfactor, upfactor)
        maze_mask = resample(maze_mask, upsampling)
        maze_ = resample(maze, upfactor)
        bg = background(canvas_shape, white, black, std, upsampling=upsampling)
    else:
        maze_ = maze
        bg = background(canvas_shape, white, black, std)

    transition = (1 - maze_mask) * bg + maze_mask * maze_

    return transition


def fade_out(seq, alpha_stop, std=0.0, black=0.0, white=1.0, downsample=None):
    """
    Make a video sequence vanish into background noise.
    :param seq: Input sequence
    :param alpha_stop: Final attenuation
    :param std: Standard deviation of the background.
    :param black: Value of the black color associated to walls.
    :param white: Value of the white color associated to cells.
    :param downsample: Downsampling factor. If set, frame size is downsampled prior to background noise generation
    and the generated background noise is consequently upsampled back to the original size.
    :return: Faded out sequence.
    """
    seq_shape = seq.shape
    seq_len = seq_shape[0]
    seq_dims = len(seq_shape)

    if downsample is None:
        bg_shape = seq_shape
        upsampling = None
    else:
        bg_shape = [seq_len, ] + [s // downsample for s in seq_shape[1:]]
        upsampling = (1,) + (downsample, ) * (seq_dims - 1)

    alpha_fadeout = alpha_stop ** (1 / seq_len)

    bg_noise = background(bg_shape, white, black, std, upsampling=upsampling)

    alpha_seq = alpha_fadeout ** np.arange(seq_len).reshape((-1,) + (1,) * (seq_dims - 1))

    fadeout = alpha_seq * seq + (1 - alpha_seq) * bg_noise

    return fadeout


def shift_video(seq, alpha_shift, alpha_t, max_shift):
    """
    Random shift of a video sequence. Shift is modeled as AR(1) along all dimensions.
    :param seq: Input sequence.
    :param alpha_shift: Parameter of AR(1) across length and width.
    :param alpha_t: Parameter of AR(1) across frames.
    :param max_shift: Maximum allowed shift.
    :return: Shifted video sequence.
    """
    n_frames, height, width = seq.shape

    std_shift = max_shift / 2 * ar_gain(alpha_t) * ar_gain(alpha_shift)

    def shift_vector(n):
        shift_vec = trunc_norm(loc=0, scale=std_shift, a=-2*std_shift, b=2*std_shift, size=(n_frames, n))
        shift_vec = ar_filter(shift_vec, alpha_t, axis=0)
        shift_vec = ar_filter(shift_vec, alpha_shift, axis=-1)
        shift_vec = np.round(clip(shift_vec, -max_shift, max_shift)).astype(np.int)
        return shift_vec

    shift_row = shift_vector(height)
    shift_col = shift_vector(width)

    shifted_seq = np.zeros_like(seq)

    for t in range(n_frames):
        frame = seq[t, ]
        for i in range(height):
            frame[i, ] = np.roll(frame[i, ], shift_row[t, i])
        for j in range(width):
            frame[:, j] = np.roll(frame[:, j], int(shift_col[t, j]))
        shifted_seq[t, ] = frame

    return shifted_seq


def insert_frames(array, n_frames, pos, ref=None, axis=0,
                  std=0.0, black=0.0, white=1.0, downsample=None):
    """
    Insert frames into the maze sequence.
    :param array: Input array containing video sequence.
    :param n_frames: Number of frames to insert.
    :param pos: Position where the frames should be inserted
    :param ref: Index reference. If set, n_frames copies of the frame at ref are inserted.
    Otherwise background noise is generate.
    :param axis: Axis into which frames are inserted
    :param std: Standard deviation of the background.
    :param black: Value of the black color associated to walls.
    :param white: Value of the white color associated to cells.
    :param downsample: Downsampling factor. If set, frame size is downsampled prior to background noise generation
    and the generated background noise is consequently upsampled back to the original size.
    :return: Array with inserted frames.
    """
    array_shape = array.shape
    ax_len = array_shape[axis]
    position = pos % ax_len

    if ref is None:
        if downsample is None:
            bg_shape = tuple(n_frames if i == axis else dim for i, dim in enumerate(array_shape))
            upsampling = None
        else:
            bg_shape = tuple(n_frames if i == axis else dim // downsample for i, dim in enumerate(array_shape))
            upsampling = tuple(1 if i == axis else downsample for i, dim in enumerate(array_shape))

        frames = background(bg_shape, white, black, std, upsampling=upsampling)

    else:
        ref_frame = np.expand_dims(array.take(ref, axis=axis), axis=axis)
        frames = np.repeat(ref_frame, n_frames, axis=axis)

    pre = array.take(range(0, position), axis=axis)
    post = array.take(range(position, ax_len), axis=axis)

    new_array = np.concatenate((pre, frames, post), axis=axis)

    return new_array


def add_wgn(array, std):
    """
    Add white Gaussian noise to array.
    :param array: Input array
    :param std: Standard deviation of Gaussian noise.
    :return: Noisy copy of array.
    """
    return array + normal(scale=std, size=array.shape)


def piecewiselin(x, y):
    """
    Create piecewise linear array
    :param x: Lengths of the segments in number of samples.
    :param y: Point values. Its length must be 1 greater than the lenght of x.
    :return: Piecewise linear array.
    """
    xlen = len(x)
    ylen = len(y)

    if (ylen - xlen) != 1:
        raise ValueError(f"Invalid dimensions len(x)={xlen}, len(y)={ylen}. len(x) must equal len(y) - 1.")

    y_start = y[:-1]
    y_stop = y[1:]

    pwl = np.array([])
    for start, stop, num in zip(y_start, y_stop, x):
        pwl = np.append(pwl, np.linspace(start, stop, num))

    return pwl
