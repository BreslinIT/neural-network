/*
This file defines the basic types for the neural network. Net is the most complex type, which uses elements from 
TrainingData and Neuron.
*/

class Neuron;

typedef std::vector<Neuron> Layer;

// Stores the weight and change of weight of the connection
// Connections act like this: thisNeuron -> nextNeuron
struct Connection {
    double weight;
    double deltaWeight;
};

class Neuron {
public:
    // Making a neuron requires the number of neurons in the next layer (minus the bias neuron)
    // (thisNeuron -> nextNeuron) for every neuron in the next layer. Neurons also need to know their position in the layer, learning
    // rate, and alpha constant.
    Neuron(const int weights, const int index, double learningRate, double alpha); 
    std::string examineWeights(void); // Returns a string in the format: "for every neuron:     outputVal
                                      //                                                        gradient
                                      //                                  for every connection: weight
                                      //                                                        deltaWeight"
    void setOutputVal(double outputVal) {m_outputVal = outputVal;}
    double getOutputVal(void) const {return m_outputVal;}
    void setGradient(double gradient) {m_gradient = gradient;}
    void resetConnections(void) {m_outputWeights.clear();}
    void addConnection(Connection weights) {m_outputWeights.push_back(weights);}
    void feedForward(Layer &prevLayer); // Sums the output of each neuron in the previous layer multiplied by the connection to this neuron. 
                                        // (prevLayer -> thisNeuron)
                                        // This neuron's output value is updated to activationFunction(sum)
    double activationFunction(double sum); // Returns tanh(sum)
    double activationFunctionDerivative(double sum); // Returns 2/(1 + cosh(2*sum)) (this is the derivative of tanh(sum))
    void calcOutputGradients(double target); // Figures out how to change the output gradient to minimize loss
    void calcHiddenGradients(Layer &nextLayer); // TODO: not sure what this is doing
    double sumDOW(const Layer &nextLayer); // Figures out how much of an impact a specific neuron had on the overall loss (I think)
    void updateInputWeights(Layer &prevLayer); // TODO: not sure what this is doing
private:
    static double randomWeight() {return rand() / double(RAND_MAX);} // Returns a number in the range [0.0 1.0]
    double m_outputVal; // The value that the neuron passes on during feedForward
    std::vector<Connection> m_outputWeights; // The length of m_outputWeights is equal to the size of the next layer -1 (for bias neuron)
    int m_index; // Neuron's position in the layer
    double m_gradient; // TODO: used in back propagation somehow
    double m_learningRate; // Used when updating weights. 0.2 is medium.
    double m_alpha; // Multiplied by the previous deltaWeight when updating weights. 0.5 is medium.
};

class TrainingData {
public:
    // Creating a TrainingData object requires a path to the training data and a path to the targets. These files must be formatted in
    // the following way (The forward slashes are necessary. When the program unpacks the data, it searches for ascii character 47):
    // trainingData
    /* 
    neuron1_input
    neuron2_input
    neuron...n_input
    /
    neuron1_input
    neuron2_input
    neuron...n_input
    /
    neuron1_input
    neuron2_input
    neuron...n_input

    */
    // trainingTargets
    /*
    target1
    target2
    target...n
    /
    target1
    target2
    target...n
    /
    target1
    target2
    target...n
    
    */

    TrainingData(const std::string dataName, const std::string targetName);
    bool isDataEOF(void) const {return m_trainingData.eof();}
    bool isTargetEOF(void) const {return m_trainingTargets.eof();}
    void populateInputs(std::vector<double> &inputs); // Interprets the formatted trainingData file and populates the passed vector
    void populateTargets(std::vector<double> &targets); // Interprets the formatted trainingTargets file and populates the passed vector
    void restart(void);
private:
    std::ifstream m_trainingData;
    std::ifstream m_trainingTargets;
};

class Net {
public:
    // To create a new network you only need to input the desired topolgy, the learning rate of each neuron, and the alpha constant of each neuron.
    Net(const std::vector<int> &topology, double learningRate, double alpha);
    void feedForward(const std::vector<double> &inputs); // Sets the output values of each neuron (except bias) in the first layer to match
                                                         // the input vector. The input vector must be the same size as the first layer.
    void backProp(const std::vector<double> &targets); // TODO: some of this makes sense. Some of it does not.
    void getResults(std::vector<double> &results) const; // Fills the passed vector with the current output values of the final layer.

    // Training takes the path of the data file and the target file, the number of training steps, the information output rate, and a
    // boolean deciding if anything is output at all.
    // It does the following steps once for every iteration:
    // It runs feedForward with the next inputs from the data file and backProp with the next targets from the target file. If either
    // of these files reaches the EOF, they are both reset.
    void train(const std::string dataName, const std::string targetName, const int iterations, const int outputRate, const bool verbose);
    void predict(const std::vector<double> &inputs, std::vector<double> &results); // Runs feedForward on the inputs and populates results vector
    double getRecentAverageError(void) const { return m_recentAverageError; }
    std::string printShape(void); // Returns the shape of the network
    std::string examineNeurons(void); // Returns the output value, the gradient, and the list of connections for each neuron
    void saveWeights(const std::string fileName); // Saves the current output values, gradients, and connections for each neuron in the network
    // Loads weights from the specified file. The file is in the following format:
    /*
    numLayers
    layer1Size
    layer2Size
    layer...nSize
    neuron1OutputVal
    neuron1Gradient
    neuron1Connection1Weight
    neuron1Connection1DeltaWeight
    neuron1Connection2Weight
    neuron1Connection2DeltaWeight
    neuron2OutputVal
    neuron2Gradient
    neuron2Connection1Weight
    neuron2Connection1DeltaWeight
    neuron2Connection2Weight
    neuron2Connection2DeltaWeight
    */
    void loadWeights(const std::string fileName);
private:
    std::vector<Layer> m_layers; // m_layers[layer][neuron]
    double m_error; // Root mean squared error
    double m_recentAverageError; // Smoothed average error
    static double m_recentAverageSmoothingFactor; // This affects the recentAverageError
    std::fstream m_saveFile;
};
