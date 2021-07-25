import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs
import copy
import random

def make_blob(num_samples, num_clusters, means, covariance):
    """
    Function to make a Synthetic Dataset based on Normal Distribution (BLOBS).
    """
    samples = None
    labels = None
    
    for i in range(num_clusters):
        # create features
        x, y = np.random.multivariate_normal(means[i], covariance[i], num_samples).T
        # join features
        f = np.column_stack((x,y))
        # set labels by cluster
        l = np.array([i]*f.shape[0])
        
        # add to X and y
        if(i == 0):
            samples = f
            labels = l
        else:
            samples = np.concatenate((samples,f), axis=0)
            labels = np.concatenate((labels, l), axis=0)
        
    return samples, labels

def noisy(samples, noise_type):
    """
    Parameters
    ----------
    image : ndarray
        Input image data. Will be converted to float.
    mode : str
        One of the following strings, selecting the type of noise to add:

        'gauss'     Gaussian-distributed additive noise.
        'poisson'   Poisson-distributed noise generated from the data.
        's&p'       Replaces random pixels with 0 or 1.
        'speckle'   Multiplicative noise using out = image + n*image,where
                    n is uniform noise with specified mean & variance.
    """
    
    if noise_type == "gauss":
        row,col = samples.shape
        mean = 0
        var = 0.05
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row,col))
        gauss = gauss.reshape(row,col)
        noisy = samples + gauss
        return noisy
    
    elif noise_type == "s&p":
        row,col = samples.shape
        amount = 0.05
        out = np.copy(samples)
        num = np.ceil(amount * samples.size)
        
        x1_min = np.amin(samples[:,0])
        x1_max = np.amax(samples[:,0])
        x2_min = np.amin(samples[:,1])
        x2_max = np.amax(samples[:,0])
                
        # Pepper mode
        coords = np.random.randint(0, samples.shape[0], int(num))
        
        for i in range(len(coords)):
            out[coords[i]][0] = random.uniform(x1_min, x1_max)
            out[coords[i]][1] = random.uniform(x2_min, x2_max)
            
        return out
    
    elif noise_type == "poisson":
        vals = len(np.unique(samples))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(samples * vals) / float(vals)
        return noisy
    
    elif noise_type =="speckle":
        row,col = samples.shape
        gauss = np.random.randn(row,col)
        gauss = gauss.reshape(row,col)
        noisy = samples + samples * gauss
        return noisy


def translation(x_t, magnitude):
    for i in range(len(x_t)):
        x_t[i][0] += magnitude
        x_t[i][1] += magnitude
    return x_t

def rotation(x_t, angle):
    # Rotate data
    # axis 1
    a = np.multiply(x_t[:,0], np.cos(angle))
    b = np.multiply(x_t[:,1], -1*np.sin(angle))
    T1 = np.add(a,b)
    # axis 2
    a = np.multiply(x_t[:,1], np.cos(angle))
    b = np.multiply(x_t[:,0], np.sin(angle))
    T2 = np.add(a,b)
    # copy rotated data
    x_t[:,0] = T1
    x_t[:,1] = T2
    return x_t

def scale(x_t, scalar):
    # Apply scale
    x_t = scalar * x_t
    return x_t

def reflection(x_t, scalar):
    # applying Reflection
    x_t[:,1] = x_t[:,1] * (-scalar)
    return x_t

def shear(x_t, scalar):
    # Applying SHEAR
    # dim-1
    a = np.multiply(x_t[:,0], 1)
    b = np.multiply(x_t[:,1], 0)
    T1 = np.add(a,b)
    # dim-2
    a = np.multiply(x_t[:,1], 1)
    b = np.multiply(x_t[:,0], scalar)
    T2 = np.add(a,b)
    # copy data
    x_t[:,0] = T1
    x_t[:,1] = T2
    return x_t

def affine_transformation(x_s, y_s, name, affine_method, value, save=True):
    """
    Method to generate affine transformation
    """
    # Copy to target domain
    x_t = copy.copy(x_s)
    y_t = copy.copy(y_s)
    # generate affine transformation
    x_t = affine_method(x_t, value)
    # Save target domain
    if save:
        np.savetxt("x_" + name + ".csv", x_t, delimiter=',')
        np.savetxt("y_" + name + ".csv", y_t, delimiter=',')
    return x_t, y_t

def show_data(x_s, y_s, x_t, y_t):
    # Draw synthetic data on image
    #x = np.concatenate((x_s, x_t), axis=0)
    #y = np.concatenate((y_s, y_t), axis=0)
    plt.clf()
    plt.scatter(x_t[:,0], x_t[:,1], c=y_t[:], alpha=0.4)
    plt.show()


# Generate BLOBS SYNTHETIC DATA
random_state = 123
centers = [(6, 6), (8, 8), (10, 6)]
samples = make_blobs(n_samples=600, centers=centers, cluster_std=[0.5, 0.5, 0.5], random_state=random_state, shuffle=True)
x_s = samples[0]
y_s = samples[1]
# save data
np.savetxt("x_dsb00.csv", x_s, delimiter=',')
np.savetxt("y_dsb00.csv", y_s, delimiter=',')

# Generate DSB01 - TRANSLATION
random_state = 1234
centers = [(6, 6), (8, 8), (10, 6)]
samples = make_blobs(n_samples=600, centers=centers, cluster_std=[0.5, 0.5, 0.5], random_state=random_state, shuffle=True)
x_s = samples[0]
y_s = samples[1]
x_t, y_t = affine_transformation(x_s, y_s, "dsb01", translation, 4)

# Generate DSB02 - SCALE
random_state = 1234
centers = [(6, 6), (8, 8), (10, 6)]
samples = make_blobs(n_samples=600, centers=centers, cluster_std=[0.5, 0.5, 0.5], random_state=random_state, shuffle=True)
x_s = samples[0]
y_s = samples[1]
x_t, y_t = affine_transformation(x_s, y_s, "dsb02", scale, 2)


# Generate DSB02 - ROTATION
random_state = 1234
centers = [(6, 6), (8, 8), (10, 6)]
samples = make_blobs(n_samples=600, centers=centers, cluster_std=[0.5, 0.5, 0.5], random_state=random_state, shuffle=True)
x_s = samples[0]
y_s = samples[1]
x_t, y_t = affine_transformation(x_s, y_s, "dsb03_15", rotation, np.pi/12)
x_t, y_t = affine_transformation(x_s, y_s, "dsb03_30", rotation, np.pi/6)
x_t, y_t = affine_transformation(x_s, y_s, "dsb03_45", rotation, np.pi/4)


# Generate DSB04 - SHEAR
random_state = 1234
centers = [(6, 6), (8, 8), (10, 6)]
samples = make_blobs(n_samples=600, centers=centers, cluster_std=[0.5, 0.5, 0.5], random_state=random_state, shuffle=True)
x_s = samples[0]
y_s = samples[1]
x_t, y_t = affine_transformation(x_s, y_s, "dsb04_5", shear, 0.5)
x_t, y_t = affine_transformation(x_s, y_s, "dsb04_10", shear, 1)
x_t, y_t = affine_transformation(x_s, y_s, "dsb04_15", shear, 1.5)

# Generate DSB05 - COMBINATION
random_state = 1234
centers = [(6, 6), (8, 8), (10, 6)]
samples = make_blobs(n_samples=600, centers=centers, cluster_std=[0.5, 0.5, 0.5], random_state=random_state, shuffle=True)
x_s = samples[0]
y_s = samples[1]
x_t, y_t = affine_transformation(x_s, y_s, "dsb05", rotation, np.pi/6)
x_t, y_t = affine_transformation(x_t, y_t, "dsb05", shear, 0.5)
x_t, y_t = affine_transformation(x_t, y_t, "dsb05", scale, 1.5)
x_t, y_t = affine_transformation(x_t, y_t, "dsb05", translation, 2)

# Generate DSB06 - SKEWED DISTRIBUTIONS
centers = [(6, 6), (10, 11), (12, 6)]
samples = make_blobs(n_samples=600, centers=centers, cluster_std=[1.0, 2.0, 0.5], random_state=random_state, shuffle=True)
x_t = samples[0]
y_t = samples[1]
x_t, y_t = affine_transformation(x_t, y_t, "dsb06", translation, 0)
show_data(x_t, y_t, x_t, y_t)


# Generate DSB07 - NOISE
random_state = 1234
centers = [(6, 6), (8, 8), (10, 6)]
samples = make_blobs(n_samples=600, centers=centers, cluster_std=[0.5, 0.5, 0.5], random_state=random_state, shuffle=True)
x_s = samples[0]
y_s = samples[1]
# noise
x_t = noisy(x_s, "s&p")
x_t, y_t = affine_transformation(x_t, y_s, "dsb07", translation, 0)
#show_data(x_s, y_s, x_t, y_t)

# Generate DSB08 - OVERLAPPING
random_state = 1234
centers = [(6, 6), (8, 8), (10, 6)]
samples = make_blobs(n_samples=600, centers=centers, cluster_std=[0.5, 0.5, 0.5], random_state=random_state, shuffle=True)
x_s = samples[0]
y_s = samples[1]
x_t = x_s + 2*x_s.std() * np.random.random(x_s.shape)
x_t, y_t = affine_transformation(x_t, y_s, "dsb08", translation, 0)
#show_data(x_s, y_s, x_t, y_t)

# Generate DSB09 - SUB-CLUSTERS
random_state = 180
centers = [(6, 6), (8, 8), (10, 6)]
samples = make_blobs(n_samples=600, centers=centers, cluster_std=[0.6, 0.6, 0.6], random_state=random_state, shuffle=True)
x_t = samples[0]
y_t = samples[1]
centers = [(7.8, 6), (6.0, 8.5)]
samples = make_blobs(n_samples=100, centers=centers, cluster_std=[0.2, 0.2], random_state=random_state, shuffle=True)
x_t = np.concatenate((x_t, samples[0]), axis=0)
y_t = np.concatenate((y_t, samples[1]), axis=0)
x_t, y_t = affine_transformation(x_t, y_t, "dsb09", translation, 0)
#show_data(x_s, y_s, x_t, y_t)





# Generate MOONS SYNTHETIC DATA
moons = make_moons(n_samples=600, noise=.1)
x_s = moons[0]
y_s = moons[1]
# save data
np.savetxt("x_dsm00.csv", x_s, delimiter=',')
np.savetxt("y_dsm00.csv", y_s, delimiter=',')

# Generate DSM01 - TRANSLATION
moons = make_moons(n_samples=600, noise=.1)
x_s = moons[0]
y_s = moons[1]
x_t, y_t = affine_transformation(x_s, y_s, "dsm01", translation, 2)

# Generate DSM02 - SCALE
moons = make_moons(n_samples=600, noise=.1)
x_s = moons[0]
y_s = moons[1]
x_t, y_t = affine_transformation(x_s, y_s, "dsm02", scale, 1.5)

# Generate DSM03 - ROTATION
moons = make_moons(n_samples=600, noise=.1)
x_s = moons[0]
y_s = moons[1]
x_t, y_t = affine_transformation(x_s, y_s, "dsm03_15", rotation, np.pi/12)
x_t, y_t = affine_transformation(x_s, y_s, "dsm03_30", rotation, np.pi/6)
x_t, y_t = affine_transformation(x_s, y_s, "dsm03_45", rotation, np.pi/4)

# Generate DSM04 - SHEAR
moons = make_moons(n_samples=600, noise=.1)
x_s = moons[0]
y_s = moons[1]
x_t, y_t = affine_transformation(x_s, y_s, "dsm04_5", shear, 0.5)
x_t, y_t = affine_transformation(x_s, y_s, "dsm04_10", shear, 1)
x_t, y_t = affine_transformation(x_s, y_s, "dsm04_15", shear, 1.5)

# Generate DSM05 - COMBINATION
moons = make_moons(n_samples=600, noise=.1)
x_s = moons[0]
y_s = moons[1]
x_t, y_t = affine_transformation(x_s, y_s, "dsm05", translation, 2)
x_t, y_t = affine_transformation(x_t, y_s, "dsm05", rotation, np.pi/4)
x_t, y_t = affine_transformation(x_t, y_s, "dsm05", scale, 2)
x_t, y_t = affine_transformation(x_t, y_s, "dsm05", shear, 1.0)

# Generate DSM06 - SKEWED DISTRIBUTION
moons = make_moons(n_samples=600, noise=.1)
x_s = moons[0]
y_s = moons[1]
# init
cls = y_s == 0
ind1 = np.squeeze(np.nonzero(cls))
ind2 = np.squeeze(np.nonzero(np.bitwise_not(cls)))
x_a = x_s[ind1]
y_a = y_s[ind1]
x_b = x_s[ind2]
y_b = y_s[ind2]
# add noise
x_a = x_a + 1.5*x_a.std() * np.random.random(x_a.shape)
x_t = np.concatenate((x_a, x_b), axis=0)
y_t = np.concatenate((y_a, y_b), axis=0)
x_t, y_t = affine_transformation(x_t, y_t, "dsm06", translation, 0)
#show_data(x_s, y_s, x_t, y_t)

# Generate DSM07 - NOISE
moons = make_moons(n_samples=600, noise=.1)
x_s = moons[0]
y_s = moons[1]
x_t = noisy(x_s, "s&p")
x_t, y_t = affine_transformation(x_t, y_s, "dsm07", translation, 0)
#show_data(x_s, y_s, x_t, y_t)


# Generate DSM08 - OVERLAPPING
moons = make_moons(n_samples=600, noise=.33)
x_t = moons[0]
y_t = moons[1]
x_t, y_t = affine_transformation(x_t, y_t, "dsm08", translation, 0)


# Generate DSM09 - COMPACT
random_state = 170
centers = [(-0.5, -0.5)]
samples = make_blobs(n_samples=100, centers=centers, cluster_std=[0.15, 0.15], random_state=random_state, shuffle=True)
moons = make_moons(n_samples=600, noise=.1)
x_t = moons[0]
y_t = moons[1]

x_t = np.concatenate((x_t, samples[0]), axis=0)
y_t = np.concatenate((y_t, samples[1]), axis=0)
x_t, y_t = affine_transformation(x_t, y_t, "dsm09", translation, 0)
#show_data(x_s, y_s, x_t, y_t)



