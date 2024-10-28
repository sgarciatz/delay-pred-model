#Author: Santiago Garcia-Gil


# import libraries
import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

import pandas as pd
import random

np.random.seed(42)
random.seed(42)
n_samples = 30000
#mean = 1200
#std_dev = 200
colors = ["khaki", "olive", "bisque", "seagreen", "steelblue", "slateblue"]

X_list = []
log_dens_list = []
smooth_Y_lists = []
bins = np.linspace(1000, 2000, 20)
X_plot = np.linspace(1000, 2000, 100)[:,np.newaxis]
for i in range(1,7):
    mean = random.randrange(1350, 1750)
    min_dev = (mean - 1000) // 1
    max_dev = (2000 - mean) // 1

    std_dev = random.randrange(0, min([min_dev, max_dev]))
    
    X = np.random.normal(mean, std_dev, n_samples)
    plt.subplot(230+i)
    #kde = KernelDensity(kernel="gaussian", bandwidth=0.5).fit(X[:])
    kde = KernelDensity(kernel="gaussian").fit(X.reshape(-1, 1))
    log_dens = kde.score_samples(X_plot)
    smooth_Y = pd.Series(log_dens).ewm(com=0.9).mean()
    X_list.append(X)
    log_dens_list.append(log_dens)
    smooth_Y_lists.append(smooth_Y)
    plt.hist(X, bins, color=colors[i-1], density=True)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
#    plt.plot(X_plot, np.exp(log_dens), color="black")
    plt.plot(X_plot,
             np.exp(smooth_Y),
             linestyle='solid',
             linewidth=2,
             color="red")
    plt.title(f"Throughput probability distribution of camera {i}",
              weight="bold")
    plt.xlabel("Throughput (Mbps)")
    plt.ylabel("Probability (%)")
    print(np.any(X < 1000))
plt.show()


# Simulate the deployment of camera 1, 2, 3, 4 and 5 and estimate the 
# probability of congestion issues
markers = ["v", "o", "P", "s", "p" ]

plt.figure(figsize=(10,6))


for i in range(5):
    probs = []
    congestion_threshold = 10000
    deployed_ms = [X_list[random.randrange(0, 5)] for _ in range(6)]
    joint_deployment = 0
    for x in deployed_ms:
        joint_deployment += x
    joint_deployment += random.randrange(0, 2) * 1000
    joint_deployment /= 1000
    samples = joint_deployment.shape[0]
    p_samples = np.where(joint_deployment > congestion_threshold,
                         1,
                         0)
    prob = np.sum(p_samples) / len(p_samples)
    bins = np.linspace(2, 12, 100)
    values = []
    for item in bins:
        p_samples = np.where(joint_deployment <= item,
                         1,
                         0)
        prob = 1 - (np.sum(p_samples) / len(p_samples))
        probs.append(prob)
        print(item)
        print(joint_deployment)
        print(f"Probability of T > {item} = {prob}%")
    plt.plot(bins,
             probs,
             marker=markers[i],
             markevery=3,
             linewidth=2,
             markersize=8)

limit = plt.axvline(x=10, 
                    color='maroon',
                    linestyle="--",
                    linewidth=4,
                    label="Maximum link throughput (φ)")
limit = plt.axhline(y=0.2, xmin =0.8, xmax=1, color = 'dodgerblue', linestyle=":", linewidth=4, label="Maximum aceptable probability of congestion")

# Axis management
plt.grid(True)
plt.xlim(2, 12)
plt.xlabel("TROUGHPUT (Gb/s)", fontsize=16)
plt.ylabel("CONGESTION PROB. (%)", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], fontsize=14)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

# Legend management
plt.legend(["Throughput CDF of Link 1",
            "Throughput CDF of Link 2",
            "Throughput CDF of Link 3",
            "Throughput CDF of Link 4",
            "Throughput CDF of Link 5",
            "Maximum link throughput ($φ_{MAX}$)",
            "Max. Acceptable prob. of Congestion (α)"],
            loc="lower left",
            fontsize=12)


#Saving fig management
plt.rcParams.update({'mathtext.default': 'regular'})
plt.tight_layout()
plt.savefig("/home/santiago/Pictures/congestion_prob.pdf",dpi=600)
plt.show()
exit()
plt.hist(joint_deployment, bins, marker=markers[i], density=True, stacked=True, cumulative=-1, histtype="step", linewidth=2)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
limit = plt.axvline(x=congestion_threshold, color = 'maroon', linestyle="--", linewidth=4, label="Available Throughput of Routing Device n")
limit = plt.axhline(y=0.2, xmin =0.8, xmax=1, color = 'steelblue', linestyle=":", linewidth=4, label="Maximum aceptable probability of congestion")
plt.xlim(2000, 12000)
plt.grid(True)
plt.title("Cumulative Distribution Function of Throughput", weight="bold")
plt.legend(["Throughput CMF of Link 1",
            "Throughput CMF of Link 2",
            "Throughput CMF of Link 3",
            "Throughput CMF of Link 4",
            "Throughput CMF of Link 5",
            "Available Throughput of Links",
            "Maximum aceptable probability of congestion"])

plt.show()
