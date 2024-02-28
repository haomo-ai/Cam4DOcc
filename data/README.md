### Data Structure

Please link your [nuScenes V1.0 full dataset ](https://www.nuscenes.org/nuscenes#download) to the data folder. 

[nuScenes-Occupancy](https://drive.google.com/file/d/1vTbgddMzUN6nLyWSsCZMb9KwihS7nPoH/view?usp=sharing), [nuscenes_occ_infos_train.pkl](https://github.com/JeffWang987/OpenOccupancy/releases/tag/train_pkl), and [nuscenes_occ_infos_val.pkl](https://github.com/JeffWang987/OpenOccupancy/releases/tag/val_pkl) are also provided by the previous work. If you only want to reproduce the forecasting results with "inflated" form, nuScenes dataset and Cam4DOcc are all you need.

Note that the folders under `cam4docc` will be generated automatically once you first run our training or evaluation scripts.

```bash
Cam4DOcc
├── data/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── lidarseg/
│   │   ├── v1.0-test/
│   │   ├── v1.0-trainval/
│   │   ├── nuscenes_occ_infos_train.pkl/
│   │   ├── nuscenes_occ_infos_val.pkl/
│   ├── nuScenes-Occupancy/
│   ├── cam4docc
│   │   ├── GMO/
│   │   │   ├── segmentation/
│   │   │   ├── instance/
│   │   │   ├── flow/
│   │   ├── MMO/
│   │   │   ├── segmentation/
│   │   │   ├── instance/
│   │   │   ├── flow/
```
The GMO folder will contain the data where vehicle and human are considered as one unified category.

The MMO folder will contain the data where vehicle and human are divided into multiple categories for clearer evaluation on forecasting performance.

In near future, we will unify GMO and MMO for easier usage.
