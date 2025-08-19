# GradPS

This code is build based on PyMARL2 and PyMARL. We assume that you have experience with PyMARL. The requirements are the same as PyMARL2.

## Run an experiment 

```shell
python src/main.py --config=qmix --env-config=stag_hunt_s # QMIX
python src/main.py --config=qmix_share --env-config=stag_hunt_s # QMIX
python src/main.py --config=qmix --env-config=sc2 with env_args.map_name=2s3z # QMIX
python src/main.py --config=qmix_share --env-config=sc2 with env_args.map_name=2s3z # GradPS-QMIX
```

The config files act as defaults for an algorithm or environment. 

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`

All results will be stored in the `Results` folder.

## Citing GradPS
If you use GradPS in your research, please cite our paper:
```tex

@inproceedings{qingradps,
  title={GradPS: Resolving Futile Neurons in Parameter Sharing Network for Multi-Agent Reinforcement Learning},
  author={Qin, Haoyuan and Liu, Zhengzhu and Lin, Chenxing and Ma, Chennan and Mei, Songzhu and Shen, Siqi and Wang, Cheng},
  booktitle={Forty-second International Conference on Machine Learning}
  year={2025}
}
```

## License

Code licensed under the Apache License v2.0
