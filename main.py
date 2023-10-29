import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def problem1():
    # Load the data
    df = pd.read_csv('data.csv')
    # Standardize the data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    # Perform PCA
    pca = PCA(2)
    principalComponents = pca.fit_transform(df_scaled)
    # Get the standard deviations of the two resulting dimensions
    std_devs = np.std(principalComponents, 0)
    print(f'Standard Deviations: {std_devs[0]:.2f}, {std_devs[1]:.2f}')


def problem2():
    # Load the data
    df = pd.read_csv('data.csv')
    # Standardize the data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    # Perform PCA
    pca = PCA(2)
    principalComponents = pca.fit_transform(df_scaled)
    # Assuming 'principalComponents' is your PCA result
    pc1 = principalComponents[:, 0]
    pc2 = principalComponents[:, 1]
    plt.scatter(pc1, pc2)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('2D Scatter Plot of PCA Result')
    plt.show()
    # result is metapoint


problem2()
