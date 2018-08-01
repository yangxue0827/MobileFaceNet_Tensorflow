# MobileFaceNet_Tensorflow    

Tensorflow implementation for [MobileFaceNet](https://arxiv.org/abs/1804.07573) which is modified from [MobileFaceNet_TF](https://github.com/xsr-ai/MobileFaceNet_TF)

## Requirements

- tensorflow >= r1.2 (support cuda 8.0, original needs tensorflow >= r1.5 and cuda 9.0)
- opencv-python
- python 3.x ( if you want to use python 2.x, somewhere in load_data function need to change, see details in comment)
- mxnet
- anaconda (recommend)

## Construction
```
├── MobileFaceNet
│   ├── arch
│       ├── img
│       ├── txt
│   ├── datasets
│       ├── faces_ms1m_112x112
│       ├── tfrecords
│   ├── losses
│   ├── nets
│   ├── output
│       ├── ckpt
│       ├── ckpt_best
│       ├── logs
│       ├── summary
│   ├── utils
```

## Datasets

1. choose one of The following links to download dataset which is provide by insightface. (Special Recommend MS1M)
* [Refined-MS1M@BaiduDrive](https://pan.baidu.com/s/1nxmSCch), [Refined-MS1M@GoogleDrive](https://drive.google.com/file/d/1XRdCt3xOw7B3saw0xUSzLRub_HI4Jbk3/view)
* [VGGFace2@BaiduDrive](https://pan.baidu.com/s/1c3KeLzy), [VGGFace2@GoogleDrive](https://drive.google.com/open?id=1KORwx_DWyIScAjD6vbo4CSRu048APoum)
2. move dataset to ${MobileFaceNet_TF_ROOT}/datasets.
3. run ${MobileFaceNet_TF_ROOT}/utils/data_process.py.

## Training

### MobileFaceNet
```
train_nets.py --max_epoch=10
              --train_batch_size=128
              --model_type=0  # mobilefacenet
```

### TinyMobileFaceNet
```
train_nets.py --max_epoch=10
              --train_batch_size=128
              --model_type=1  # tinymobilefacenet
```

## Inference

### MobileFaceNet
```
python inference.py --pretrained_model='./output/ckpt_best/mobilefacenet_best_ckpt'
                    --model_type=0
```

### TinyMobileFaceNet
```
python inference.py --pretrained_model='./output/ckpt_best/tinymobilefacenet_best_ckpt'
                    --model_type=1
```

## Performance

### [Original result](https://github.com/xsr-ai/MobileFaceNet_TF)
|  size  | LFW(%) | Val@1e-3(%) | inference@MSM8976(ms) |
| ------ | ------ | ----------- | --------------------- |
|  5.7M  | 99.25+ |    96.8+    |          260-         |

### My training results
| Models | LFW | Cfp_FF | Cfp_FP | Agedb_30 | inference@i7-7700 16G 240G (fps) |
|------------|:---:|:--:|:--:|:--:|:--:|
|MobileFaceNet(Bad training)|0.983+-0.008|0.980+-0.005|0.827+-0.019|0.878+-0.023|27|
|Tiny_MobileFaceNet|0.981+-0.008|0.984+-0.006|0.835+-0.019|0.882+-0.023|50|

## References

1. [facenet](https://github.com/davidsandberg/facenet)
2. [InsightFace mxnet](https://github.com/deepinsight/insightface)
3. [InsightFace_TF](https://github.com/auroua/InsightFace_TF)
4. [MobileFaceNets: Efficient CNNs for Accurate Real-Time Face Verification on Mobile Devices](https://arxiv.org/abs/1804.07573)
5. [CosFace: Large Margin Cosine Loss for Deep Face Recognition](https://arxiv.org/abs/1801.09414)
6. [InsightFace : Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)
7. [tensorflow-triplet-loss](https://github.com/omoindrot/tensorflow-triplet-loss)
8. [MobileFaceNet_TF](https://github.com/xsr-ai/MobileFaceNet_TF)
