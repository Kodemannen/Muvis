import sys

import numpy as np
from scipy.fft import fft, ifft
import scipy.signal as signal

import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns
sns.set()

import pydub
import pydub.playback

import librosa

import moviepy.editor as mpe

from moviepy.video.tools.drawing import color_split

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

    fig, ax = plt.subplots()
    image = original_image.copy() / 255

    dump_path = path + "Million_dollar_anvil/"
    T = timesteps.shape[0]
    f = 10/T
    w = 2*np.pi*f

    image_shape = image.shape
    Nx = image_shape[0]
    Ny = image_shape[1]
    stk = 1000

    def update_frame(t):

        print(t/T *100, "%")        # printing progress
        ax.clear()

        x_indices = np.random.randint(low=0, high=Nx, size=stk)
        y_indices = np.random.randint(low=0, high=Ny, size=stk)

        image[x_indices,y_indices] = 0

        # Normalize:
        #image -= image.min()
        #image /= image.max()

        ax.axis("off")
        ax.imshow(image)

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist="Me"), bitrate=800)

    ani = animation.FuncAnimation(fig, update_frame, timesteps)
    ani.save("test.mp4", writer=writer)

    return 0



#-------------------------------------------------------------------#
'|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||'
#-------------------------------------------------------------------#

def main():

    song = pydub.AudioSegment.from_mp3("million_dollar_anvil.mp3")
    frame_rate = song.frame_rate

    #pydub.playback.play(song)

    song_array = np.array(song.get_array_of_samples())
    song_array = song_array.reshape((-1, 2))
    song_array = song_array / 2**15               # Normalize

    mono = song_array[:,0]
    mono = mono/1000 # in units of s now

    templength = round(len(mono)/25)

    song_duration_sec = (mono.shape[0]-1)/frame_rate
    song_duration_min = song_duration_sec/60

    fps = 24
    N_timesteps = round(song_duration_sec * fps)
    N_timesteps /= 25  # for development
    timesteps = np.arange(N_timesteps)

    img = plt.imread("me.jpg") # values between 0 and 255 it seems

    anim = animate(img, timesteps, fps)


    #plt.plot(timesteps,mono)
    #plt.savefig("tester.jpg")

    exit("hore")

    # importing image:




    #img = (img > 255/3) * img

    #plt.imsave("tester.jpg", img)



    #-------------------------------------------
    # Analysis:

    # 1. Finding the important frequencies:

    #librosa.display.specshow()
    #tempogram = librosa.feature.tempogram(mono)
    #print(tempogram)

    #print(tempogram.shape)


if __name__=="__main__":
    main()
