# vittm-hmr
Vision Token Turing Machine using hierarchical memory reconstruction SSL pretext for computational pathology.

# create_data_subset.py: 
Searches dir for .svs WSI and randomly selects 'n' slides. Generates low-res tissue mask for each slide and calculates coordinates for patches that meet a tissue density threshold. Finaly aggregates slide IDs and coords of all valid tissue patches into a csv. 

# create_data_subset_full.py
Scales logic from create_data_subset.py to process entire dataset. Uses multiproccesing to distribute workload. It assigns slides to different worker processes. Finaly aggregates slide IDs and coords of all valid tissue patches into a csv. 

# dataloader.py
Applies custom transform (ECT) that extracts a slightly larger reigon than neccessary and randomly crops it to target size, introducing spatial variance. For each slide the dataset retrieves the patch, applies normalisation, and returns a dictionary containing two identical tensor views of the image.

# evaluate.py
Evaluates a pre-trained ViTTM-HMR model on an external dataset by freezing the weights and training a linear classifier on the extracted features to distinguish between tumor / non-tumor. 

# label_tcga.py 
Extracts labels from slide id and adds column to manifest csv. NOTE: Labels not used for training - they were used in an old attempt to evaluate the model. 

# train_full.py
Initiates full-scale training run of ViTTM-HMR. It uses high performance loading configurations to work with the very large TCGA dataset. It is optimised for GPU. Script saves checkpoints after every epoch.

# train_full.py
Initates a trial-run of ViTTM-HMR. The logic is similar, but lacks some of the optimisation found in train_full.py

# model.py
Implements ViTTM-HMR. It augments a standard ViT with an external learnable memory bank that remains the same accross all layers, allowing network to store and rertieve 'prototype' features. Inside each block, there is a three-step cycle. READ: image tokens gather context from memory using linear-cross attention. COMPUTE: image tokens process information using standard self-attention. WRITE: memory clots update themselves based on the processed image tokens. The model is trained using a SSL objective, model takes a masked, low-res version of the origional slide and tries to reproduce the origional image, minimising cosine distance between the predicted and target image. 

# verify_data.py
Check dataset has downloaded correctly.