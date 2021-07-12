import cv2
import numpy as np

'''
Pour installer opencv:
sudo apt-get install opencv* python3-opencv


Si vous avez des problèmes de performances, vous pouvez calculer les convolutions plus rapidement avec :

lateral = signal.fftconvolve(activation, kernel, mode='same')

/!\ Votre kernel doit être de taille impaire pour que la convolution fonctionne correctement (taille_dnf * 2 - 1 par exemple).
'''

images_path = ""  # TODO: placez le chemin vers les images.
image_size = (380, 455)
window_pos = (50, 280)
window_size = (150, 150)


# Pour Opencv : Hue [0,179], Saturation [0,255], Value [0,255]
def selectByColor(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # TODO: créez une carte de saillance en sélectionnant uniquement le jaune
    # TODO: normalisez ce masque (valeurs entre [0,1])
    return frame


def findCenter(potentials):
    # TODO: calculez le centre de gravité de la bulle d'activation du dnf
    pass


def moveWindow(center, speed):
    # TODO: déplacez graduellement la fenêtre d'attention pour placer le centre de gravité du dnf au centre de celle-ci
    pass


def track(frame):
    input = selectByColor(frame)
    # dnf.input = input
    # dnf.update_map()
    # cv2.imshow("Input", dnf.input)
    # cv2.imshow("Potentials", dnf.potentials)
    # center = findCenter(dnf.potentials)
    # moveWindow(center, speed)


if __name__ == '__main__':
    frame = cv2.imread(images_path+"pacman00001.png")
    # TODO : initialisez votre DNF ici

    for i in range(1, 196):
        frame = cv2.imread(images_path + "pacman{0:05d}.png".format(i))
        track(frame)
        frame_np = np.asarray(frame)
        window = frame_np[window_pos[1]:window_pos[1]+window_size[1], window_pos[0]:window_pos[0]+window_size[0]]
        cv2.imshow("Input", window)
        key = cv2.waitKey(500)
        if key == 27:  # exit on ESC
            break

    cv2.destroyAllWindows()
