import cv2
import numpy as np
from scipy import ndimage



'''
Pour installer opencv:
sudo apt-get install opencv* python3-opencv


Si vous avez des problèmes de performances, vous pouvez calculer les convolutions plus rapidement avec :

lateral = signal.fftconvolve(activation, kernel, mode='same')

/!\ Votre kernel doit être de taille impaire pour que la convolution fonctionne correctement (taille_dnf * 2 - 1 par exemple).
'''

images_path = "/users/yabernar/workspace/watSOM/Data/pacman_cropped/"
image_size = (380, 455)
window_pos = (50, 280)
window_size = (150, 150)

# Pour Opencv : Hue [0,179], Saturation [0,255], Value [0,255]
def selectByColor(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([25,150,150])
    upper_yellow = np.array([45,255,255])
    # Threshold the HSV image to get only yellow colors
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    selection = np.asarray(mask)
    selection = np.divide(selection, 255) #(480, 640)
    return selection


def findCenter():
    center = ndimage.measurements.center_of_mass(dnf.potentials)
    return center[0]/size[0], center[1]/size[1]


def moveWindow(center):
    pass

def track(frame):
    input = selectByColor(frame)
    # dnf.input = input
    # dnf.update_map()
    # cv2.imshow("Input", dnf.input)
    # cv2.imshow("Potentials", dnf.potentials)
    # center = findCenter()
    # motorControl(center)


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
