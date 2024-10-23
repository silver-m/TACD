import numpy as np
import pandas as pd
from skimage import io
import torch
import glob
from torch.utils.data.dataset import Dataset

class MyDataset(Dataset):
    def __init__(self, csv_path, image_ids, patch_size, nb_dates):  # , transform

        self.data_info = pd.read_csv(csv_path)

        self.patch_size = patch_size
        self.nb_dates = nb_dates
        # self.transform = transform  # 这个

        self.all_imgs = []       # list40 每个为18 1024 1024 4
        for fold in image_ids:
            all_tifs = glob.glob(fold + '/images/*.tif')   # *.tif*

            years = []
            for j in range(len(all_tifs)):
              ff = all_tifs[j].find('monthly_')
              years.append(all_tifs[j][ff+8:ff+12] + all_tifs[j][ff+13:ff+15])
            ind = np.argsort(years)
            sort_tifs = [all_tifs[i] for i in ind]          
            img = []

            #create list with the multi-temporal images
            #you can change the step in the for loop to adjust the number of dates you want to exploit
            #Here, I have put a step=2 which creates a list with the first available date, the last available date and 8 intermediate (10 dates)
            for nd in range(0, nb_dates-1, 1):    # 0479E有问题，删掉了两张图像
                im = io.imread(sort_tifs[nd])
                img.append(im)
            img.append(io.imread(sort_tifs[-1]))
            self.all_imgs.append(np.asarray(img))
            # aa=io.imread('/data/dataset-common/myin-data/SpaceNet7vs/train/L15-1615E-1206N_6460_3366_13/images/global_monthly_2017_07_mosaic_L15-1615E-1206N_6460_3366_13.tif')
        #change groundtruth
        self.all_labels = []
        for fold in image_ids:
            label = io.imread(fold + '/change/change.tif')
            self.all_labels.append(label)

        #buildings groundtruth
        self.all_buildings = []      # list40 每个为2 1024 1024     # list40 每个为1024 1024
        for fold in image_ids:
            buildings = []
            buildings.append(io.imread(fold + '/buildings/buildings1.tif'))
            buildings.append(io.imread(fold + '/buildings/buildings2.tif'))
            self.all_buildings.append(np.array(buildings))

        # Calculate len
        self.data_len = self.data_info.shape[0]-1

    def __getitem__(self, index):
        x = int(self.data_info.iloc[:,0][index])
        y = int(self.data_info.iloc[:,1][index])
        image_id = int(self.data_info.iloc[:,2][index])
        find_patch = self.all_imgs[image_id] [:, x:x + self.patch_size, y:y + self.patch_size, :]
        find_patch = find_patch/255.0 #normalize patch

        find_labels = self.all_labels[image_id] [x:x + self.patch_size, y:y + self.patch_size]

        find_builds = self.all_buildings[image_id] [:,x:x + self.patch_size, y:y + self.patch_size]
        # find_patch = np.ascontiguousarray(find_patch)
        # find_labels = np.ascontiguousarray(find_labels)
        # find_builds = np.ascontiguousarray(find_builds)
        # sample = {'img': find_patch, 'label': find_labels, 'builds': find_builds}
        # if self.transform is not None:
        #     sample_train = self.transform(sample)  # must convert to torch tensor

        # return sample_train
        return np.ascontiguousarray(find_patch), np.ascontiguousarray(find_labels), np.ascontiguousarray(find_builds)

    def __len__(self):
        return self.data_len


