import matplotlib.pyplot as plt

from suplabel import suplabel


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    for idx, axis in enumerate(axes.ravel(), start=1):
        axis.set_title(f"title{idx}")
        axis.set_xlabel(f"xlabel{idx}")
        axis.set_ylabel(f"ylabel{idx}")
    suplabel(fig, "super X label", "x")
    suplabel(fig, "super Y label", "y")
    suplabel(fig, "super Title", "t")
    plt.show()


if __name__ == "__main__":
    main()
