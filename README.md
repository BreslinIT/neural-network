# neural-network
This is a neural network that I made with the help of https://www.youtube.com/watch?v=sK9AbJ4P8ao. The original video is from David Miller. I started with the basics from the video and expanded upon them.

## things I have added:
- created training/test data input and output
- implemented saving and loading network weights
- added some stuff for looking at the network in depth
- some python scripting for generating training data and targets
- a way to visually represent the network
- make an easier interface for choosing constants such as learning rate (maybe improve this)
- add an animation to the network visualizer
![gif of network training for 2000 iterations](images/2000.gif?raw=true)

## things I want to add:
- improve overall documentation
- create multiple activation functions
- batch size
- create a gui for training and testing
- make a deep Q network

## how to make it work
### training your network:
1. Change the values in testData.txt and testTargets.txt to represent your input data. Each value should be on a new line, with every batch seperated by a '/'.
2. Open network.cpp and change the topology values. These are the number of neurons in each layer of your network.
3. Set your training parameters.
4. Compile network.cpp. Using g++, this looks like: `g++ network.cpp -o network -lstdc++fs` (you may need to link other things)
5. Run `./network > output.txt`

### displaying your network:
1. Make sure that you saved your network weights after training.
2. Run `python3 displayNetwork.py`
#### to create a gif:
1. Make sure that you enabled the createGif parameter during training. (This defaults to false)
2. Run `python3 createGif.py`
