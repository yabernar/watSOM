import os
import subprocess


def generate_background(folder_path, nb_images):
    bkg_path = os.path.join(folder_path, "bkg.jpg")
    if os.path.isfile(bkg_path):  # remove background if it already exists
        os.remove(bkg_path)
    list_ps = subprocess.Popen('ls', stdout=subprocess.PIPE, cwd=folder_path)
    write_ps = subprocess.Popen(('head', '-{}'.format(nb_images)), stdin=list_ps.stdout, stdout=subprocess.PIPE, cwd=folder_path)
    subprocess.run(('>', 'selected_images.txt'), stdin=write_ps.stdout, cwd=folder_path)

    # subprocess.run(["ls", "|", "head", "-{}".format(nb_images), ">", "selected_images.txt"], cwd=folder_path, shell=True)  # select the nb first images from
    # the folder
    subprocess.run(["convert", "@selected_images.txt", "-evaluate-sequence", "median", "bkg.jpg"], cwd=folder_path, shell=True)  # generating background from
    # the
    # median
    os.remove(os.path.join(folder_path, "selected_images.txt"))  #cleaning up

    # COMMAND LINE
    # ls | head -100 > tmp.txt; convert @tmp.txt -evaluate-sequence median bkg.jpg; rm tmp.txt


if __name__ == '__main__':
    CDNET_path = "/users/yabernar/GrosDisque/CDNET14/dataset"
    generate_background(os.path.join(CDNET_path, "nightVideos", "fluidHighway", "input"), 100)


