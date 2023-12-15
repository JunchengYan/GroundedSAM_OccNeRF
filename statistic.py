from nuscenes.nuscenes import NuScenes
import pickle
from torch.utils.data import Dataset, DataLoader
import os, sys
import numpy as np
import concurrent.futures as futures

## For calculating the proportion of each class of the generated semantic segmentation labels
class Generator(object):
    def __init__(self, split='train'):
        self.data_path = 'data/nuscenes/nuscenes_semantic'
        version = 'v1.0-trainval'
        self.nusc = NuScenes(version=version,
                            dataroot='data/nuscenes', verbose=True)

        with open(f'data/nuscenes/nuscenes_infos_{split}.pkl', 'rb') as f:
            self.data = pickle.load(f)['infos']

        self.camera_names = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_FRONT_RIGHT']
        self.value_counts = {}

    def __call__(self, num_workers=64):
        print('generating nuscene depth maps from LiDAR projections')

        def process_one_sample(index_data):
            index_t = self.data[index_data]['token']
            rec = self.nusc.get('sample', index_t)
            
            for cam in self.camera_names:
                camera_sample = self.nusc.get('sample_data', rec['data'][cam])
                # load image
                mask = np.fromfile(os.path.join(self.data_path, camera_sample['filename'][:-4] + '_mask.bin'), dtype=np.int8).reshape(900, 1600)
                # print(mask.shape)
                # print(mask.max())
                # print(mask.min())
                unique_values, counts = np.unique(mask, return_counts=True)
                for value, count in zip(unique_values, counts):
                    self.value_counts[value] = self.value_counts.get(value, 0) + count
            
            if index_data % 1000==0:
                total_elements = sum(self.value_counts.values())
                for value, count in self.value_counts.items():
                    proportion = count / total_elements
                    print(f"Value {value}: Times {count}, Percentage {proportion:.2%}")

            print('finish processing index = {:06d}'.format(index_data))

        sample_id_list = list(range(len(self.data)))
        with futures.ThreadPoolExecutor(num_workers) as executor:
            executor.map(process_one_sample, sample_id_list, timeout=10)


if __name__ == "__main__":
    model = Generator('train')
    model()
    print("\n==================TOTAL====================")
    total_elements = sum(model.value_counts.values())
    for value, count in model.value_counts.items():
        proportion = count / total_elements
        print(f"数值 {value}: 出现次数 {count}, 占比 {proportion:.2%}")
