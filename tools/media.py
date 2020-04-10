from pathlib import Path
import subprocess
import warnings

import matplotlib.pyplot as plt
import matplotlib.animation as anim
from scipy.io.wavfile import write

from tools import signal_processing as sp

fps_default = 24

media_dir = Path.cwd() / 'media'


def time2samples(t, rate):
    """
    Convert time to number of samples at a certain rate.
    :param t: Time in seconds
    :param rate: Sampling rate.
    :return: Number of samples for the given time at the given rate.
    """
    return int(round(t * rate))


def frameshow(frame, v=None, cmap='gist_gray'):
    """
    Plot one frame
    :param frame: Frame to plot.
    :param cmap: Colour mapping used by matplotlib.
    :param v: Tuple with lower and upper values for the colour axis. If None, use the min and max values
    :return: Figure and image objects
    """
    f = plt.figure()

    if v is None:
        vmin = frame.min()
        vmax = frame.max()
    else:
        vmin = v[0]
        vmax = v[1]

    im = plt.imshow(frame, vmin=vmin, vmax=vmax, cmap=cmap)

    aspect_ratio = frame.shape[1] / frame.shape[0]

    margin = 0.85

    plt.axis('off')
    plt.tight_layout(pad=0)
    w, h = f.get_size_inches()
    f.set_size_inches([margin * w, margin * w / aspect_ratio])

    return f, im


def animation(frames, fps=fps_default, resolution=None, cmap='gist_gray', v=None, filename=None):
    """
    Create (and save) an animation from a frame array
    :param frames: Frame array.
    :param fps: Frames per second.
    :param resolution: Video resolution expressed as a number, e.g. 1080
    :param cmap: Colour mapping used by matplotlib.
    :param v: Tuple with lower and upper values for the colour axis. If None, use the min and max for the last frame.
    :param filename: Filename to save the animation. Accepted extensions are mp4 and gif.
    If None, show animation without saving.
    :return: None.
    """

    f, im = frameshow(frames[-1, ], v=v, cmap=cmap)

    if resolution is None:
        dpi = None
    else:
        w = min(f.get_size_inches())
        dpi = resolution / w

    def update(num, data, img):
        img.set_data(data[num, ])
        return img,

    im_ani = anim.FuncAnimation(f, update, frames.shape[0], fargs=(frames, im), interval=1000 / fps, blit=True)

    if filename is None:
        plt.show()
    else:

        savepath = media_dir / filename
        ext = savepath.suffix

        mp4 = '.mp4'
        gif = '.gif'

        if ext == mp4:
            writer = anim.writers['ffmpeg']
        elif ext == gif:
            writer = anim.writers['imagemagick']
        else:
            raise ValueError(f"{ext} files are not supported. Try using {mp4} or {gif} instead")

        im_ani.save(str(savepath), dpi=dpi, writer=writer(fps=fps, metadata=dict(artist='Me'), bitrate=-1))


def audio(adsr_times, adsr_levels, filename, n_ch=2, f_c=8000, fs=44100):
    """
    Generate and save noise audio according to an Attack-Decay-Sustain-Release curve
    :param adsr_times: Times of the diferent stages in seconds. The order is (pre, attack, decay, sustain, release).
    :param adsr_levels: dB levels of the ADSR curve, as (dB Floor, dB Peak, dB Sustain)
    :param filename: Filename to save the audio
    :param n_ch: Number of channels. Default is 2
    :param f_c: Cutoff frequency for the filtering. Default is 8 kHz.
    :param fs: Sampling frequency. Default is 44.1 kHz
    :return: None
    """
    n_pre, n_attack, n_decay, n_sustain, n_release = (time2samples(t, fs) for t in adsr_times)
    db_floor, db_sustain, db_peak = adsr_levels

    if max(adsr_levels) > -6.0:
        raise warnings.warn(f"Your dB levels are {adsr_levels}. Levels greater than -6 dB may saturate.", UserWarning)

    # dynamic_range = 40
    # sigma_final = 0.5

    #################################################################
    #                                                               #
    #                       ADSR Curve                              #
    # dB Level ^                                                    #
    #          |                                                    #
    #    Peak  |                X                                   #
    #          |               X XX                                 #
    #          |              X    XX                               #
    # Sustain  |             X       XXXXXXXXXXXXX                  #
    #          |            X                     XX                #
    #          |           X                        XX              #
    #          |          X                           XX            #
    #          |        XX                              XX          #
    #   Floor  |XXXXXXXXX                                 X         #
    #          +--------------------------------------------> Time  #
    #          |-------|-------|-----|------------|---------|       #
    #             Pre   Attack  Decay   Sustain     Release         #
    #################################################################

    adsr_curve = sp.piecewiselin((n_pre, n_attack, n_decay, n_sustain, n_release),
                                 (db_floor, db_floor, db_peak, db_sustain, db_sustain, db_floor))
    magnitudes = sp.db2mag(adsr_curve)

    aud_sig = sp.normal(loc=0.0, scale=1.0, size=(magnitudes.size, n_ch))
    aud_sig *= magnitudes[..., None]

    alpha_aud = sp.get_alpha(3.0, fs / f_c)
    audio_filt = sp.ar_filter(aud_sig, alpha_aud, axis=0)
    audio_clip = sp.clip(audio_filt, -1.0, 1.0)

    savepath = media_dir / filename
    write(savepath, fs, audio_clip)


def merge(animation_fname, audio_fname, output_fname):
    """
    Merge audio and video
    :param animation_fname: Filename for the animation
    :param audio_fname: Filename for the audio
    :param output_fname: Filename for the output.
    :return: Return code from ffmpeg.
    """

    vid = media_dir / animation_fname
    aud = media_dir / audio_fname
    output = media_dir / output_fname

    cmd = f'ffmpeg -y -i {vid} -i {aud} -c:v copy -c:a aac -strict experimental {output}'
    return subprocess.call(cmd, shell=True)
