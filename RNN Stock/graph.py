import matplotlib.pyplot as plt

def showAndSaveGraph(title: str, data: [([float], str)], fileName: str):
    plt.title(title)
    for d in data:
        plt.plot(d[0], label=d[1])
    plt.grid(True)
    plt.legend()
    plt.savefig(fileName)
    plt.show()
