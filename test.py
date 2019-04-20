import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import ffm


def main():
    img = mpimg.imread('blank.png')
    plt.figure(figsize=(10, 10))
    plt.imshow(img)

    x = plt.ginput(1)[0]
    x = (int(round(x[1])), int(round(x[0])))
    ffm.fmm(img, x, field=0.75)
    plt.imshow(img)

    plt.show()


if __name__ == "__main__":
    main()
