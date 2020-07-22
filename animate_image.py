import sys

import numpy as np

#import scipy
#from scipy.fft import fft, ifft
#import scipy.signal as signal

import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib
matplotlib.use('agg')
#import seaborn as sns
#sns.set()

#import pydub
#import pydub.playback

import librosa

#import moviepy.editor as mpe

#from moviepy.video.tools.drawing import color_split

# Importer coverbilde
# fuck med et filter som varierer med frekvensene i laata


path = "/media/kodemannen/C7C9-58F8/Muvis_data/"

def read_mp3(f, normalized=False):
    """MP3 to numpy array"""
    a = pydub.AudioSegment.from_mp3(f)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
    if normalized:
        return a.frame_rate, np.float32(y) / 2**15
    else:
        return a.frame_rate, y


#-------------------------------------------------------------------#
'|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||'
#-------------------------------------------------------------------#
def playit():

    # Stack and synchronize between beat events
    # This time, we'll use the mean value (default) instead of median
    beat_mfcc_delta = librosa.util.sync(np.vstack([mfcc, mfcc_delta]), beat_frames)



def animate_image():

    filename = "million_dollar_anvil.mp3"

    y, sr = librosa.load(filename)      # y is a numpy array, sr = sample rate

    # cutting for developing:
    #cut = int(y.shape[0]/10)
    #y = y[:cut]

    song_duration = (y.shape[0]-1)/sr   # sec


    # Set the hop length; at 22050 Hz, 512 samples ~= 23ms
    hop_length = 512


    # separating percussion and tones
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)
    # beat_frames contain the indices of the frames in y that are the beat hits

    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    # beat times are now beat events in seconds


    # Making animation:
    original_image = plt.imread("me.jpg") # values between 0 and 255 it seems
    fig, ax = plt.subplots()
    image = original_image.copy() / 255
    black_image = np.zeros_like(image)
    fps = 24

    #timesteps = np.arange(y.shape[0])
    timesteps_grid = np.arange(0, song_duration, step=1/fps)
    print(timesteps_grid.shape)

    def check_if_on_grid(val, grid, tol):

        cond = abs(grid - val) < tol 
        cond2 = abs(val - grid) < tol
        return cond*cond2


    tol = 2e-2

    beat_times_gridded = np.zeros_like(timesteps_grid)
    n = beat_times.shape[0]
    for i in range(n):
       
        val = beat_times[i]
        truth = check_if_on_grid(val, timesteps_grid, tol=tol)
        beat_times_gridded += truth


    dump_path = path + "Million_dollar_anvil/"
    #T = timesteps.shape[0]
    timesteps = np.arange(timesteps_grid.shape[0])

    duration = 10       # duration of black img
    count = 0
    fig.savefig('hmmm.png')

    def update_frame(i):
        
        ax.clear()
        if beat_times_gridded[i] == 0:
            ax.imshow(original_image)
        else:
            ax.imshow(black_image)

        ax.set_axis_off()
        return 0

    import time    
    t0 = time.time()

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist="Me"), bitrate=800)

    ani = animation.FuncAnimation(fig, update_frame, timesteps)   #, fargs=(count,indices))
    ani.save("exports/million_dollar_anvil.mp4", writer=writer)

    t1 = time.time() - t0
    print('time spent ', t1, ' s')
    print('time spent ', t1/60, ' min')
    return 0


def combine_with_sound():

    import moviepy.editor as mpe
    movieclip = mpe.VideoFileClip('exports/million_dollar_anvil.mp4')
    soundtrack = mpe.AudioFileClip('million_dollar_anvil.mp3')

    #final_audio = mpe.CompositeAudioClip([soundtrack])
    final_clip = movieclip.set_audio(soundtrack)

    final_clip.write_videofile('exports/test2.mp4', fps=24, codec='mpeg4')


    
def mhmovie():
    #from mhmovie.code import movie, music
    #m = movie("exports/million_dollar_anvil.mp4")
    #mu = music('million_dollar_anvil.mp3')
    #final = m+mu
    #final.save("test2.mp4")
    pass


if __name__=="__main__":
    animate_image()
    combine_with_sound()
    print('doneit')
