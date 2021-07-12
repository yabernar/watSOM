import os
import numpy as np
from PIL import Image, ImageOps, ImageChops
from Code.Parameters import Parameters, Variable
from Code.SOM import SOM
from Data.Mosaic_Image import MosaicImage

pictures_dim = [10, 10]
name = "office"
path = os.path.join("Data", "spikes", "office", "src")
bkg = Image.open(os.path.join(path, name+"0.png"))
img_parameters = Parameters({"pictures_dim": pictures_dim})
data = MosaicImage(bkg, img_parameters)
nb_epochs = 100
parameters = Parameters({"alpha": Variable(start=0.6, end=0.05, nb_steps=nb_epochs),
                         "sigma": Variable(start=0.5, end=0.001, nb_steps=nb_epochs),
                         "data": data.get_data(),
                         "neurons_nbr": (10, 10),
                         "epochs_nbr": nb_epochs})
map = SOM(parameters)
for i in range(nb_epochs):
    print("Epoch " + str(i + 1))
    map.run_epoch()

map.data = data.get_data()

output_path = os.path.join("Data", "spikes", "office", "out")
os.makedirs(os.path.join(output_path, "difference"), exist_ok=True)
os.makedirs(os.path.join(output_path, "diff_winners"), exist_ok=True)
os.makedirs(os.path.join(output_path, "saliency"), exist_ok=True)
os.makedirs(os.path.join(output_path, "thresholded"), exist_ok=True)

initial_map = map.get_all_winners()
for i in range(828):
    print("Extracting ", i)
    current = Image.open(os.path.join(path, name+str(i)+".png"))
    new_data = MosaicImage(current, img_parameters)
    map.set_data(new_data.get_data())
    winners = map.get_all_winners()
    diff_winners = map.get_neural_distances(initial_map, winners)
    diff_winners -= 0
    diff_winners[diff_winners < 0] = 0
    diff_winners = diff_winners.reshape(new_data.nb_pictures)
    diff_winners = np.kron(diff_winners, np.ones((pictures_dim[0], pictures_dim[1])))
    # diff_winners *= 30  # Use this parameter ?
    diff_winners = ImageOps.autocontrast(Image.fromarray(diff_winners).convert('L'))

    reconstructed = Image.fromarray(new_data.reconstruct(map.get_reconstructed_data(winners)))
    som_difference = ImageOps.autocontrast(ImageChops.difference(reconstructed, current).convert('L'))
    som_difference_modulated = ImageChops.multiply(som_difference, diff_winners)

    # Binarizing
    fn = lambda x: 255 if x > 10 else 0
    thresholded = som_difference_modulated.convert('L').point(fn, mode='1')

    result = Image.new("L", current.size)
    result.paste(thresholded, (0, 0))

    som_difference_modulated = ImageOps.autocontrast(som_difference_modulated)
    som_difference.save(os.path.join(output_path, "difference", "dif"+str(i)+".png"))
    diff_winners.save(os.path.join(output_path, "diff_winners", "win"+str(i)+".png"))
    som_difference_modulated.save(os.path.join(output_path, "saliency", "sal"+str(i)+".png"))
    thresholded.save(os.path.join(output_path, "thresholded", "thr"+str(i)+".png"))