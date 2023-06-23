import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class ProperOrthogonalDecomposition:
    def __init__(self, path: str):
        self.path = path

    def run(self):
        roi = pd.read_csv(self.path)
        x = StandardScaler().fit_transform(roi)

        pca = PCA()
        principalComponents = pca.fit_transform(x)

        explained_variance = pca.explained_variance_ratio_

        cumulative_explained_variance = np.cumsum(explained_variance)

        print(cumulative_explained_variance)

        num_components = np.where(cumulative_explained_variance > 0.95)[0][0]

        pca = PCA(n_components=num_components)
        principalComponents = pca.fit_transform(x)

        np.savetxt("PrincipalComponents.csv", principalComponents, delimiter=",")


if __name__ == "__main__":
    try:
        file = input("Please input the .csv file you want to analyze\n")
        pod = ProperOrthogonalDecomposition(file)
        pod.run()
    except FileNotFoundError:
        print("Error, File not found")
