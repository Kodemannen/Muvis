import numpy as np
import pydub
import sys
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.fft import fft, ifft
import scipy.signal as signal
import seaborn as sns
sns.set()

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



def write(f, sr, x, normalized=False):
    """numpy array to MP3"""
    channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
    if normalized:  # normalized array - each item should be a float in [-1, 1)
        y = np.int16(x * 2 ** 15)
    else:
        y = np.int16(x)
    song = pydub.AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
    song.export(f, format="mp3", bitrate="320k")


def Traveling_point(track, peaks_x, peaks_y, frame_rate):
    # pydub works in ms
    n = len(track)
    T = (n-1)/frame_rate # s
    dt = 1/frame_rate
    fps = 30

    t = np.linspace(0, T, n)
    k = 10 # Using the k first freqs


    freqs = peaks_x[:k] / 1000   # Hz
    print(freqs)
    amps  = peaks_y[:k]

    f = freqs[0]
    w = 2*np.pi*f
    a = amps[0]
    s = np.sin(w*t)

    fig, ax = plt.subplots()

      ##################
     # Trailing dots: #
    ##################
    n_dots = 10
    rgba_colors = np.zeros((n_dots, 4))
    rgba_colors[:,1] = 1.0
    alphas = np.linspace(0.1, 1, n_dots)
    rgba_colors[:,3] = alphas # 4th columns are the alphas

    # Some numerical integration method:
    #v0 = np.array([1,0])
    #v = np.zeros((n,2))
    #r = np.zeros((n,2))
    #for i in range(n):


    def update_data(i):
        print(t/T *100, "%")
        i = int(frame_rate/fps*i) # Makes sure the time is correct
        ax.clear()
        ax.set_ylim([-1.5,1.5])

        step = 80
        reachback = step*n_dots # the farthest backwards reaching trailing dot

        tall = len(s[(i-reachback):i:step])
        ax.scatter(t[(i-reachback):i:step],s[(i-reachback):i:step],color=rgba_colors[:tall])
        ax.set_title("t={lok:.1f} s".format(lok=t[i]))

    ani = animation.FuncAnimation(fig=fig, func=update_data)

    Writer = animation.writers['ffmpeg']

    writer = Writer(fps=fps, metadata=dict(artist="Me"), bitrate=800)
    ani.save('traveling_point.mp4', writer=writer)
    exit("eddge")


def get_freqs(track)


def main():
    filename = 'million_dollar_anvil.mp3'





if __name__ == "__main__":
    #import inspect
    # track = read('perc_prosjekt.mp3')
    #from pydub import AudioSegment
    #from pydub.playback import play

    #sound = AudioSegment.from_file("perc_prosjekt.mp3", format="mp3")
    #play(sound)

    a = pydub.AudioSegment.from_mp3("perc_prosjekt.mp3")
    frame_rate = a.frame_rate

    track = np.array(a.get_array_of_samples())
    track = track.reshape((-1, 2))
    track = track / 2**15               # Normalize

    mono = track[:,0]
    track = mono
    track = track/1000 # in units of s now

    # Fourier:
    freqs = fft(track) # Hz
    peaks = signal.find_peaks(x=freqs, height=4)

    #print("n peaks: ", len(peaks[0]))

    peaks_x = peaks[0]
    peaks_y = peaks[1]["peak_heights"]

    Traveling_point(track, peaks_x, peaks_y, frame_rate)

    fig, ax = plt.subplots()
    ax.plot(freqs)
    ax.scatter(peaks_x, peaks_y, color="black")

    plt.semilogx()
    fig.savefig("dev_fig.pdf")
    #plt.show()
    # cvlc traveling_point.mp4 --input-repeat 1999
    # vlc traveling_point.mp4 --input-repeat 1999
