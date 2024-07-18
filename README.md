# A cross-attention-based deep learning approach for predicting functional stroke outcomes using 4D CTP imaging and clinical metadata
Submission for Medical Image Analysis, MEDIA-D-24-00425

</div>

## Abstract
Acute ischemic stroke (AIS) remains a global health challenge, leading to long-term functional disabilities without timely intervention. Spatio-temporal (4D) Computed Tomography Perfusion (CTP) imaging is crucial for diagnosing and treating AIS due to its ability to rapidly assess the ischemic core and penumbra. \snm{Although traditionally used to assess acute tissue status in clinical settings, 4D CTP has also been explored in research for predicting stroke tissue outcomes. However, its potential for predicting functional outcomes, especially in combination with clinical metadata, remains unexplored.} Thus, this work aims to develop and evaluate a novel multimodal deep learning model for predicting functional outcomes (specifically, 90-day modified Rankin Scale) in AIS patients by combining 4D CTP and clinical metadata. To achieve this, an intermediate fusion strategy with a cross-attention mechanism is introduced to enable a selective focus on the most relevant features and patterns from both modalities. Evaluated on a dataset comprising 70 AIS patients who underwent endovascular mechanical thrombectomy, the proposed model achieves an accuracy (ACC) of 0.771, outperforming conventional late fusion strategies (ACC=0.729) and unimodal models based on either 4D CTP (ACC=0.614) or clinical metadata (ACC=0.714). The results demonstrate the superior capability of the proposed model to leverage complex inter-modal relationships, emphasizing the value of advanced multimodal fusion techniques for predicting functional AIS outcomes.

<p align="center">
<img src="https://github.com/kimberly-amador/Multimodal-Stroke-Outcome-Prediction/blob/main/figs/model_architecture_v2.png" width="750">
</p>


## Usage

#### Installation

Recommended environment:

- Python 3.8.1
- TensorFlow GPU 2.4.1
- CUDA 11.0.2 
- cuDNN 8.0.4.30

To install the dependencies, run:

```shell
$ git clone https://github.com/kimberly-amador/Multimodal-Stroke-Outcome-Prediction
$ cd Multimodal-Stroke-Outcome-Prediction
$ pip install -r requirements.txt
```

#### Data Preparation
1. Preprocess the data. The default model takes images of size 512 x 512 x 16.
2. Save the preprocessed images and their corresponding labels as numpy arrays into a single file in 'patientID_preprocessed.npz' format. 
3. Create a patient dictionary. This should be a pickle file containing a dict as follows:

```python
partition = {
    'train': {
        'patientID1',
        'patientID2',
        ...
    },
    'val': {
        'patientID3',
        'patientID4',
        ...
    }
    'test': {
        'patientID5',
        'patientID6',
        ...
    }
}
```

#### Train Model

1. Modify the model configuration. The default configuration parameters are in `./model/config_file.py`.
2. Run `python main.py` to train the model.
