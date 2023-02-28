In this repository, I made the first attempts to learn the embeddings of impulse responses. 
I used 2 impulse response databases:

-	ARNI (alto university, varying positions of acoustic panels)
-	BUT (Brno university, real rooms recordings)

For each impulse responses using matlab toolbox I computed room acoustic parameters: 
-	Rt (reveberation time) 
-	Drr (direct to reverberant ratio)
-	Edt (early decay time) 
-	Cte (early-to-late index)

The idea is to present impulse responses in the database to an autoencoder network and make the network learn the embedding space which best represents impulse. To validate if the network has learned meaningful parameters, I use the visualization method TSNE, which reduces the embeddings to a 2-dimensional space, which can be plotted. If in the plots, the clusters correspond to known similarities in the acoustic parameters – good. 

I used 3 autoencoder networks (to be found in models.py): 

-	Variational autoencoder with MLP layers, which expects waveform IR at the input
-	Convolutional variational autoencoder, which also expects a waveform IR
-	Convolutional autoencoder (non-var), which expects a normalized log spectrogram

There are following notebooks in this repository: 

-	datacheck&training.ipynb : reading the data, some basic plots representing parameters of impulse responses across the data bases, and the training procedure
-	vae_ir_embeddings.ipynb: plotting embeddings of the variational autoencoders in the tsne space
-	ae_ir_embeddings.ipynb: plotting embeddings of the linear autoencoder in the tsne space
