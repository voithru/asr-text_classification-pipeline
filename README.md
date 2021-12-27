# ASR and Text_classification Pipeline

Text classification with Speech Recognition pipeline

- We used Wav2Vec for speech recognition. If you want to know how to finetune wav2vec, please see [here](https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/Fine_tuning_Wav2Vec2_for_English_ASR.ipynb)(for korean, see [here](https://github.com/voithru/wav2vec2_finetune))

- We used Electra(especially KoElectra as we worked on Korean dataset) for text classification. If you want to know how to finetune electra, please see here: [for Korean](https://github.com/monologg/KoELECTRA)

## Installation

```commandline
pip install -r requirements.txt
```

## Inference

Assume you have both wav2vec and electra.

```commandline
python main.py [-h] [--wav_dir WAV_DIR] [--stt_output_path STT_OUTPUT_PATH] [--output_path OUTPUT_PATH] [--wav2vec_checkpoint WAV2VEC_CHECKPOINT] [--electra_checkpoint ELECTRA_CHECKPOINT]

optional arguments:
  -h, --help            show this help message and exit
  --wav_dir WAV_DIR
  --stt_output_path STT_OUTPUT_PATH
  --output_path OUTPUT_PATH
  --wav2vec_checkpoint WAV2VEC_CHECKPOINT
  --electra_checkpoint ELECTRA_CHECKPOIN
```

The STT results will be in `--stt_output_path`. Final predicted output will be in `--output_path`.
