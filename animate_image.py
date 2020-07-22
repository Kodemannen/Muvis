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


def animate(original_image, timesteps, fps):
    """OLD SHIT"""

    fig, ax = plt.subplots()
    image = original_image.copy() / 255
    image_shape = image.shape

    dump_path = path + "Million_dollar_anvil/"
    T = timesteps.shape[0]
    f = 10/T
    w = 2*np.pi*f

    count = 0

    def update_frame(t, count):

        count += 1
        #-------------------------------------------
        print(t/T *100, "%") # Printing progress and
        ax.clear()           # clearing canvas
        #-------------------------------------

        # Normalize:
        #image -= image.min()
        #image /= image.max()

        ax.axis("off")
        ax.imshow(image)

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist="Me"), bitrate=800)

    ani = animation.FuncAnimation(fig, update_frame, timesteps, fargs=(count,))
    ani.save("test.mp4", writer=writer)

    return 0


def animate_image():

    filename = "million_dollar_anvil.mp3"

    y, sr = librosa.load(filename)      # y is a numpy array, sr = sample rate

    # cutting for developing:
    cut = int(y.shape[0]/10)
    y = y[:cut]

    song_duration = (y.shape[0]-1)/sr   # sec


    # Set the hop length; at 22050 Hz, 512 samples ~= 23ms
    hop_length = 512


    # separating percussion and tones
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    # beat_frames contain the indices of the frames in y that are the beat hits


    # Compute MFCC features from the raw signal
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
    # returns a matrix with shape (n_mfcc, T) where T is the track duration in frames
    print(mfcc.shape)


    # And the first-order differences (delta features)
    mfcc_delta = librosa.feature.delta(mfcc)


    # Stack and synchronize between beat events
    # This time, we'll use the mean value (default) instead of median
    beat_mfcc_delta = librosa.util.sync(np.vstack([mfcc, mfcc_delta]),
                                        beat_frames)

    # Compute chroma features from the harmonic signal
    chromagram = librosa.feature.chroma_cqt(y=y_harmonic,
                                            sr=sr)

    # Aggregate chroma features between beat events
    # We'll use the median value of each feature between beat frames
    beat_chroma = librosa.util.sync(chromagram,
                                    beat_frames,
                                    aggregate=np.median)

    # Finally, stack all beat-synchronous features together
    beat_features = np.vstack([beat_chroma, beat_mfcc_delta])
    # contains indices of beat frames
    # shape: (38, 473), meaning (features, event_frames)

    print(beat_features.shape) 
    
    print(beat_features[0,0])

    # Making animation:
    original_image = plt.imread("me.jpg") # values between 0 and 255 it seems
    fig, ax = plt.subplots()
    image = original_image.copy() / 255
    black_image = np.zeros_like(image)
    fps = 24

    timesteps = np.arange(y.shape[0])
    timesteps = np.arange(100)
    dump_path = path + "Million_dollar_anvil/"
    T = timesteps.shape[0]
    indices = beat_features[0]


    duration = 10       # duration of black img
    count = 0

    def update_frame(t, count, indices):

        ax.imshow(original_image)


        return 0

        

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist="Me"), bitrate=800)

    ani = animation.FuncAnimation(fig, update_frame, timesteps, fargs=(count,indices))
    ani.save("test.mp4", writer=writer)

    return 0



#-------------------------------------------------------------------#
'|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||'
#-------------------------------------------------------------------#
def playit():

    # Stack and synchronize between beat events
    # This time, we'll use the mean value (default) instead of median
    beat_mfcc_delta = librosa.util.sync(np.vstack([mfcc, mfcc_delta]), beat_frames)

def main():
    """OLD SHIT"""
    song = pydub.AudioSegment.from_mp3("million_dollar_anvil.mp3")
    frame_rate = song.frame_rate

    #pydub.playback.play(song)

    song_array = np.array(song.get_array_of_samples())
    song_array = song_array.reshape((-1, 2))
    song_array = song_array / 2**15               # Normalize

    mono = song_array[:,0]
    mono = mono/1000 # in units of s now

    templength = round(len(mono)/25)
    mono = mono[:templength]

    song_duration_sec = (mono.shape[0]-1)/frame_rate
    song_duration_min = song_duration_sec/60

    fps = 24
    N_timesteps = round(song_duration_sec * fps)
    N_timesteps = int(N_timesteps)

    timesteps = np.arange(N_timesteps)

    tempogram = librosa.feature.tempogram(y=mono)
    tempogram = np.mean(tempogram, axis=1)


    peak_indices = scipy.signal.find_peaks(tempogram)[0]

    plt.plot(tempogram)
    plt.scatter(peak_indices, tempogram[peak_indices])
    #plt.plot(peaks)
    plt.savefig("develop_fig.jpg")

    img = plt.imread("me.jpg") # values between 0 and 255 it seems

    anim = animate(img, timesteps, fps)



if __name__=="__main__":
    animate_image()
