from tools import signal_processing as sp, media


im_name = 'wnm_animation.mp4'
audio_name = 'white_nightmare.wav'
video_name = 'white_nightmare.mp4'

maze_width = 64
maze_height = 64
wall_thickness = 4

noise_std = 0.4
background_std = .25
white = 0.5
black = -0.5

fps = media.fps_default

sec_fadein = 1
sec_start = 1.5 - sec_fadein
sec_sustain = 3 - sec_fadein
sec_fadeout = 2

start_frames = media.time2samples(sec_start, fps)
fadein_frames = media.time2samples(sec_fadein, fps)
sustain_frames = media.time2samples(sec_sustain, fps)
end_frames = media.time2samples(sec_fadeout, fps)

max_shift = wall_thickness * 1.5
lag = wall_thickness * 4
att = sp.mag2db(2 * max_shift)

alpha_maze = .5      # Do not set higher than .5
alpha_stop = .1
alpha_t = .8
alpha_shift = sp.get_alpha(att, lag)

bg_args = {'std': background_std, 'black': black, 'white': white}  # Parameters for background generation

if __name__ == "__main__":

    print("Generating maze graph.")
    root, children = sp.wilson_algorithm(maze_height, maze_width)
    print("Rendering basic maze.")
    maze, maze_history = sp.render_maze(root, children, alpha_maze, **bg_args)

    sec_maze = maze_history.shape[0] / fps
    total_sec = sec_start + sec_fadein + sec_maze + sec_sustain + sec_fadeout
    print(f"Building the maze takes {round(sec_maze, 1)} seconds @{fps}fps."
          f"Total duration is {round(total_sec, 1)} seconds.")

    print("Rendering transition from noise to maze.")
    canvas = sp.render_transition(maze, maze_history, fadein_frames, alpha_stop,
                                  **bg_args, upfactor=wall_thickness)
    print("Rendering transition from maze to noise.")
    canvas = sp.insert_frames(canvas, sustain_frames + end_frames, pos=-1, ref=-1)
    canvas[-end_frames:, ] = sp.fade_out(canvas[-end_frames:, ],
                                         alpha_stop, downsample=wall_thickness, **bg_args)
    print("Prepending noise frames.")
    canvas = sp.insert_frames(canvas, start_frames, pos=0, ref=None,
                              downsample=wall_thickness, **bg_args)
    print("Shifting maze.")
    canvas = sp.shift_video(canvas, alpha_shift, alpha_t, max_shift)
    print("Adding fine-grain noise.")
    canvas = sp.add_wgn(canvas, noise_std)

    print("Plotting/saving animation.")
    media.animation(canvas, filename=im_name)
    print("Saving audio.")
    media.audio(adsr_times=(sec_start, sec_maze, sec_fadein, sec_sustain, sec_fadeout),
                adsr_levels=(-50, -6, -9), filename=audio_name)
    print("Merging animation and audio")
    media.merge(im_name, audio_name, video_name)
    print("Video merged.")
