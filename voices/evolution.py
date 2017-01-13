import os

import numpy as np
import matplotlib.pyplot as plt

from lasp.sound import WavFile, plot_spectrogram
from lasp.timefreq import gaussian_stft
from voices.motor_programs import LinearMotorProgram
from voices.oscillator import TwoMassOscillator


class CrossEntropyEvolver(object):

    def __init__(self, population_size, initial_mean, initial_covariance):
        self.num_params = len(initial_mean)
        self.mean = initial_mean
        self.covariance = initial_covariance
        self.resample_population(population_size)

    def resample_population(self, num_samps):
        self.population =  np.random.multivariate_normal(self.mean, self.covariance, size=num_samps)

    def run_population(self, num_seconds=1., num_renditions=5, dt=20e-5, output_dir=None):
        nagents,nparams = self.population.shape

        nt = int(num_seconds / dt)

        all_renditions = np.zeros([nagents, num_renditions, nt])
        all_controls = np.zeros([nagents, num_renditions, 2, nt])
        for k,p in enumerate(self.population):

            # initialize the two mass oscillator parameters
            m = p[0]
            k0 = p[1]
            k12 = p[2]
            eta = p[3]
            print 'm=%0.3f, k0=%d, k12=%d, eta=%0.6f' % (m, k0, k12, eta)

            # create several renditions of a song using a parameterized motor program
            for j in range(num_renditions):
                print 'Simulating rendition %d of agent %d' % (j, k)
                tmo = TwoMassOscillator(m=m, k0=k0, k12=k12, eta=eta)
                mp = LinearMotorProgram()
                mp.set_params(p[4:])

                for t in range(nt):
                    [u, dk] = mp.next()
                    tmo.step_rk4(dk, u, dt)
                    all_controls[k, j, :, t] = [u, dk]
                    all_renditions[k, j, t] = tmo.x[0]

        plt.figure()
        gs = plt.GridSpec(5, 3)
        for k in range(nagents):
            for j in range(num_renditions):
                ax = plt.subplot(gs[k, j])
                # plt.plot(all_renditions[k, j, :], '-')
                spec_t, spec_freq, spec, rms = gaussian_stft(all_renditions[k, j, :], 1. / dt, 20e-3, 1e-3,
                                                             min_freq=0, max_freq=5e3)
                plot_spectrogram(spec_t, spec_freq, spec, ax=ax, colorbar=False, ticks=False)
                plt.axis('tight')
                plt.xticks([])

        plt.figure()
        gs = plt.GridSpec(5, 3)
        for k in range(nagents):
            for j in range(num_renditions):
                ax = plt.subplot(gs[k, j])
                plt.plot(all_controls[k, j, 0, :], 'k-')
                plt.axis('tight')
                plt.xticks([])
                plt.ylim(0, 200)
        plt.suptitle('Control u')

        plt.figure()
        gs = plt.GridSpec(5, 3)
        for k in range(nagents):
            for j in range(num_renditions):
                ax = plt.subplot(gs[k, j])
                plt.plot(all_controls[k, j, 1, :], 'r-')
                plt.axis('tight')
                plt.xticks([])
                plt.ylim(-1e3, 1e3)
        plt.suptitle('Control k')

        plt.show()

        if output_dir is not None:
            for k in range(nagents):
                for j in range(num_renditions):
                    ofile = os.path.join(output_dir, 'agent_%d_rendition%d.wav' % (k, j))
                    wf = WavFile()
                    wf.sample_rate = 1. / dt
                    wf.data = all_renditions[k, j, :]
                    wf.num_channels = 1
                    try:
                        wf.to_wav(ofile, normalize=True)
                    except:
                        print 'Problem with rendition %d of agent %d' % (j, k)


if __name__ == '__main__':

    num_params = 4 + 16 + 8
    mean = np.zeros([num_params])
    mean[0] = 0.250 # mass
    mean[1] = 1e6 # k0
    mean[2] = 30 # k12
    mean[3] = 1e-5 # eta
    mean[4:] = 1.0

    cov = np.eye(num_params)*1.0
    # cov[0, 0] = 0.050
    # cov[1, 1] = 1e2
    # cov[2, 2] = 1e1
    # cov[3, 3] = 1e-1
    cov[0, 0] = 0.0
    cov[1, 1] = 0.0
    cov[2, 2] = 0.0
    cov[3, 3] = 0.0

    cee = CrossEntropyEvolver(5, mean, cov)
    cee.run_population(num_renditions=3, output_dir='/tmp/renditions')
