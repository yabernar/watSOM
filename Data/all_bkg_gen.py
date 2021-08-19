import os
import subprocess


if __name__ == '__main__':
    cdnet_path = "/home/yabernar/watSOM/Data/tracking/dataset/"
    cdnet_path = "/home/yabe/Code/watSOM/Data/tracking/dataset/"
    exclusion_list = ["PTZ"]

    categories = sorted([d for d in os.listdir(cdnet_path) if os.path.isdir(os.path.join(cdnet_path, d))], key=str.lower)
    for cat in categories:
        if cat not in exclusion_list:
            elements = sorted([d for d in os.listdir(os.path.join(cdnet_path, cat)) if os.path.isdir(os.path.join(cdnet_path, cat, d))], key=str.lower)
            for elem in elements:
                print(os.path.join(cdnet_path, cat, elem))
                os.system("cd "+os.path.join(cdnet_path, cat, elem, "input")+'; ulimit -n 2000; vips bandrank "$(echo in0000*.jpg)" ../back.jpg')
                #os.system("cd "+os.path.join(cdnet_path, cat, elem)+'; convert ROI.bmp ROI.png')




