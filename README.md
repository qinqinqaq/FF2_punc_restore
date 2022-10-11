# FF2_punc_restore

## Methods
+ parallel_enc_dec.py(model fusion)
+ modeling_electra.py(interaction attention)

## Directory
+ **train.py** - Training Process
+ **config.py** - Training Configurations
+ **res/data/raw** - IWSLT Source Data
+ **src/models** - Models
+ **src/utils** - Helper Function

## Setup
Please ensure required packages are already installed. A virtual environment is recommended.
```
$ cd FF2_punc_restore
$ pip install pip --upgrade
$ pip install -r requirements.txt
```

## Run
Before training, please take a look at the **config.py** to ensure training configurations.
```
$ vim config.py
$ python train.py
```

## Output
If everything goes well, you should see a similar progressing shown as below.
```
*Configuration*
Initialize...
cloudy-yellow-ferret
Some weights of ElectraModel were not initialized from the model checkpoint at google/electra-large-discriminator and are newly initialized: ['electra.encoder.layer.0.attention.self.talk_matrix', 'electra.encoder.layer.1.attention.self.talk_matrix', 'electra.encoder.layer.2.attention.self.talk_matrix', 'electra.encoder.layer.3.attention.self.talk_matrix', 'electra.encoder.layer.4.attention.self.talk_matrix', 'electra.encoder.layer.5.attention.self.talk_matrix', 'electra.encoder.layer.6.attention.self.talk_matrix', 'electra.encoder.layer.7.attention.self.talk_matrix', 'electra.encoder.layer.8.attention.self.talk_matrix', 'electra.encoder.layer.9.attention.self.talk_matrix', 'electra.encoder.layer.10.attention.self.talk_matrix', 'electra.encoder.layer.11.attention.self.talk_matrix', 'electra.encoder.layer.12.attention.self.talk_matrix', 'electra.encoder.layer.13.attention.self.talk_matrix', 'electra.encoder.layer.14.attention.self.talk_matrix', 'electra.encoder.layer.15.attention.self.talk_matrix', 'electra.encoder.layer.16.attention.self.talk_matrix', 'electra.encoder.layer.17.attention.self.talk_matrix', 'electra.encoder.layer.18.attention.self.talk_matrix', 'electra.encoder.layer.19.attention.self.talk_matrix', 'electra.encoder.layer.20.attention.self.talk_matrix', 'electra.encoder.layer.21.attention.self.talk_matrix', 'electra.encoder.layer.22.attention.self.talk_matrix', 'electra.encoder.layer.23.attention.self.talk_matrix']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
num_parameters: 442582788

*Configuration*
model: parallelendecoder
language model: google/electra-large-discriminator
freeze language model: False
sequence boundary sampling: random
mask loss: False
trainable parameters: 442,582,788
model:
encode_layer.embeddings.position_ids	torch.Size([1, 512])
encode_layer.embeddings.word_embeddings.weight	torch.Size([30522, 1024])
encode_layer.embeddings.position_embeddings.weight	torch.Size([512, 1024])
encode_layer.embeddings.token_type_embeddings.weight	torch.Size([2, 1024])
encode_layer.embeddings.LayerNorm.weight	torch.Size([1024])
encode_layer.embeddings.LayerNorm.bias	torch.Size([1024])
encode_layer.encoder.layer.0.attention.self.talk_matrix	torch.Size([16, 16])
encode_layer.encoder.layer.0.attention.self.query.weight	torch.Size([1024, 1024])
encode_layer.encoder.layer.0.attention.self.query.bias	torch.Size([1024])
encode_layer.encoder.layer.0.attention.self.key.weight	torch.Size([1024, 1024])
encode_layer.encoder.layer.0.attention.self.key.bias	torch.Size([1024])
encode_layer.encoder.layer.0.attention.self.value.weight	torch.Size([1024, 1024])
encode_layer.encoder.layer.0.attention.self.value.bias	torch.Size([1024])
encode_layer.encoder.layer.0.attention.output.dense.weight	torch.Size([1024, 1024])
encode_layer.encoder.layer.0.attention.output.dense.bias	torch.Size([1024])
encode_layer.encoder.layer.0.attention.output.LayerNorm.weight	torch.Size([1024])
encode_layer.encoder.layer.0.attention.output.LayerNorm.bias	torch.Size([1024])
encode_layer.encoder.layer.0.intermediate.dense.weight	torch.Size([4096, 1024])
encode_layer.encoder.layer.0.intermediate.dense.bias	torch.Size([4096])
encode_layer.encoder.layer.0.output.dense.weight	torch.Size([1024, 4096])
encode_layer.encoder.layer.0.output.dense.bias	torch.Size([1024])
encode_layer.encoder.layer.0.output.LayerNorm.weight	torch.Size([1024])
encode_layer.encoder.layer.0.output.LayerNorm.bias	torch.Size([1024])
...
device: cuda
train size: 8950
val size: 1273
ref test size: 54
asr test size: 54
batch size: 4
train batch: 2237
val batch: 319
ref test batch: 14
asr test batch: 14
valid win size: 8
if load check point: False


Training...
  0%|          | 0/2237 [00:00<?, ?it/s]
Loss:1.5445:   0%|          | 0/2237 [00:00<?, ?it/s]                                                                                                                                                                                                 | 5/559 [00:02<03:30,  2.63it/s]
```

## Authors
* **Kebin Fang** -fkb@zjuici.com

