This project develops a new spatial attention mechanism for relevant text detection.

## Datasets required to be downloaded
* [DeepFashion2 Dataset](https://github.com/switchablenorms/DeepFashion2)
* [ICDAR15](https://rrc.cvc.uab.es/?ch=4)

## Instruction for Training
* Modify the `config.py` file.
    * Line 17: enter directory path of DeepFashion2 Dataset Images
    * Line 18: enter directory path of DeepFashion2 Dataset Annotations (in JSON format)
    * Line 23: enter directory path of ICDAR15 trainset path
    * Line 24: enter directory path of ICDAR15 testset path
    * Pray to the Almightly that your `config.py` is safely configured.
    * `python main.py` (you can also specify which GPU to use by `CUDA_DEVICE_VISIBLE=2 python main.py`)

## Instruction for Testing
* Go figure it our yourself by looking into `predict.py` :p

## Ping me at:
contact@pinakinathc.me

## Check other kool stuff at:
www.pinakinathc.me

## Thanks to:
* Sauradip Nag (The Engineer)
* Dr. P. Shivakumara (The Main Locomotive driving this work)
* Dr. R. Raghavendra (The Auxillary Locomotive Enhancing this work)
* Dr. Umapada Pal (The Station Master)
* Me: The passenger :p
