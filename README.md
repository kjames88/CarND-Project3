##Network Architecture

I started from the Nvidia Dave-2 architecture from the paper.  I tried modifications on this but I either hit memory limitations or did not improve accuracy.  Aside from adding dropout and max pooling to limit overfitting, the final architecture closely resembles Nvidia's.  Following the lectures I used relu for activations and softmax for the final classification.  _However, it would turn out that the problems were much more related to training data than to reported accuracy numbers._

### Layer Detail

1. BatchNormalization
2. Convolution2D, nb_filter=24, nb_row=5, nb_col=5, subsample=(2,2)
3. MaxPooling2D, pool_size=(2,2), strides=(1,1)
4. ELU
5. Dropout, 0.25
6. Convolution2D, nb_filter=36, nb_row=5, nb_col=5, subsample=(2,2)
7. ELU
8. Dropout, 0.25
9. Convolution2D, nb_filter=48, nb_row=5, nb_col=5, subsample=(2,2)
10. ELU
11. Convolution2D, nb_filter=64, nb_row=3, nb_col=3
12. ELU
13. Convolution2D, nb_filter=64, nb_row=3, nb_col=3
14. ELU
15. Flatten
16. Dense, 100 (a) or 1000 (b)
17. ELU
18. Dropout, 0.5
19. Dense, 50 (a) or 500 (b)
20. ELU
21. Dense, 21 (a) or 1 (b)
22. Softmax (a only)


To summarize, the model performs normalization of inputs, followed by five convolutional layers, two flat layers, and final classification layer.  The first three convolutional layers use 5x5 filters with (2,2) stride, while the remaining convolutional layers uses 3x3 filters with (1,1) stride.  The first layer uses 2x2 max pooling and 0.25 dropout.  The next layers uses 0.25 dropout.  After the convolutional layers, results are flattened and fed into a series of three dense layers, the final of which yields the output.  ELU activation is used in each layer except the final layer.  ELU provided slightly better accuracy convergence than ReLU.

### Categorical Angle Method

The classification layers of the model are two dense layers of size 100 (dropout 0.5) and 50, followed by a final dense layer of size 21 (steering angle classes).  The final layer activation is softmax.  An Adam optimizer with _sparse_categorical_crossentropy_ objective is used for training.

With Keras, I observed floating point results, while I expected one-hot output based on previous labs.  I used max of the outputs to decide the classification but this may not be the same as previous behavior.

### Integer Angle MSE Method

The final layers of the model are dense of size 1000 (dropout 0.5) and 500, followed by a final layer of size 1 (steering angle as a number rather than a class).  There is on softmax activation; instead the output is used as-is (linear activation).  An Adam optimizer with _mse_ (mean_squared_error) objective is used for training.

I used larger dense layers in this case because I read advice online to increase the number of neurons with MSE to avoid numerical stability issues.  In fact, I found that the loss would sometimes increase dramatically (e.g. spike from 3.5 to 600) even with larger layers, although that change seemed to improve the situation.

![Drawing of Architecture]
(./images/arch.png)

##Training

It took many passes to train the network to the point that it (mostly) stopped driving onto the shoulder or off into the water.  I used a joystick to make the input smoother (for me the mouse did not work well at all in linux).  Initially I completed laps that seemed fairly clean, trained the network, and promply went off the road.  Later I added failure cases with the car facing the side of the lane at an angle, driving along the shoulder, aiming itself for the water, etc.  Observing the autonomous driving at each step clearly demonstrates the difficulty of providing enough views to enable recovery from each action.  Further, the physics complicates matters since the correct steering angle is subtle:  turning hard on the inside shoulder around a curve vs turning a little too hard and driving over the shoulder.  One thing I tried was using the side camera images, with the logic that the steering angle is the same given a lateral shift.  I think this confused the network (side views vs error-recovery training images) and my outcome improved after I dropped the side views.

The following images show a few views in the error recovery around a turn.  To prevent the car from driving straight over the shoulder into the water, multiple sequences were recorded showing the view either tracking closely to the right shoulder, or facing too sharply toward the left shoulder, wheels turned right, following around and away from the left shoulder and the water.

### Correction Images at a Problem Curve

![Image of Right Shoulder]
(./images/center_2017_01_29_20_10_17_090.jpg)

![Image Facing Left Shoulder]
(./images/center_2017_01_29_20_14_15_018.jpg)

![Image Approaching Left Shoulder]
(./images/center_2017_01_29_20_14_41_040.jpg)

### Categorical Angle Method

I augmented the data by generating shifted or rotated versions of the center camera images.  This allows the network to learn very similar views that may be encountered in live runs, thus improving its prediction accuracy when tested.  Input images are windowed to focus on the useful region rather than the hood of the car or the scenery above the road.  While the network should learn which features are important for prediction, there is no purpose in using additional processing resources to discover what is already known.

The majority of my datasets are stacked together, shuffled, and run through training.  To refine the performance around curves, escaping from lane markings, etc, several additional overlay training sets were applied.  I used one epoch of training on top of a good candidate network, and all layers were updated during the training.  In previous attempts I found that the latest data was overweighted, so I reduced to a single epoch in order to prevent the new data from having undue influence on the prior weights.

During training, for this project I found training accuracy to be a poor metric.  Whether the car gets around the track is a matter of trial and error with the training data selection.  I'm not including learning rate visualization for that reason, but the outcome was generally near 0.82 accuracy with the main training data and validation data (20%) having relatively similar accuracy.

### Integer Angle MSE Method

Because the categorical method below tended to produce a _twitchy_ model, I switched to mean squared error optimization on integer inputs.  (Floating point steering angle values were converted to/from integer steps).  As for the categorical method, input images are windowed to focus on the useful region.  Using MSE I trained on five epochs with the captured training images without augmentation.  I also used a python generator to supply training and validation images for this method.  Attempts to add small data sets to tune the model did not perform well, and the final results were obtained by training on the full data set in one pass.

##Changes From First Submission

### First Round

In my original training, I had driven as if on a track, making use of, or at least not worrying about, the shoulders.  Since the requirements include avoiding the shoulders, I started over with new training runs.  Doing so also appears to have removed some poor training data that resulted from my skills with the joystick, and the second iteration generally went much more smoothly than the first.  On the reviewer's suggestion, I also switched from ReLU to ELU activation.  This showed some improvement in accuracy, and may have helped with the driving behavior as a result.  Further, I added a data set captured from the alternate test track.  It may be coincidence since I only did this once, but the resulting network was much less twitchy than its predecessor (this training was one epoch overlayed on the predecessor).

### Second Round

I switched to MSE and added a python generator.