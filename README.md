-------------------------------- PHASE 2: --------------------------------

The next quite natural extension of the PHASE 1 is to try to estimate the impulse response from the reverberant signal. 

Motivation: In VR/XR it might help to change the room acoustic properties of sound recorded in one space, so that it sounds as if it came from another space. This problem can be devided into two sub-tasks where the first one is to de-reverberate a signal and the second one is to convolve the de-reverberated signal with a new room impulse response to obtain a perceptually different space. However, the desired room impulse response is usually not explicitly given, so a part of the problem is to estimate a room impulse response from a reverberant signal recorded in a target acoustic space. 

Task: Given a mono, reverberant signal, estimate the room impulse response. 

Approach: Use deep learning to solve that problem. Use encoder-decoder architecture with the waveform at the input and room impulse response at the output. 

Data: Speech convolved with various impulse responses. 

TODOS: 
- Add synthetic IRs to the list of available IRS
- Define architecture(s)
- Training 


-------------------------------- PHASE 1 (last commit Feb 28, 2023): --------------------------------

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
