class Neuron;

typedef std::vector<Neuron> Layer;

struct Connection {
    double weight;
    double deltaWeight;
}; // stores the weight and change of weight of the connection

class Neuron {
public:
    Neuron(const int weights, const int index);
    std::string examineWeights(void); // prints the weight values and output value
    void setOutputVal(double outputVal) {m_outputVal = outputVal;}
    double getOutputVal(void) const {return m_outputVal;}
    void setGradient(double gradient) {m_gradient = gradient;}
    void resetConnections(void) {m_outputWeights.clear();}
    void addConnection(Connection weights) {m_outputWeights.push_back(weights);}
    void feedForward(Layer &prevLayer);
    double activationFunction(double sum);
    double activationFunctionDerivative(double sum);
    void calcOutputGradients(double target);
    void calcHiddenGradients(Layer &nextLayer);
    double sumDOW(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);
private:
    static double randomWeight() {return rand() / double(RAND_MAX);} // returns a number in the range [0.0 1.0]
    double m_outputVal;
    std::vector<Connection> m_outputWeights;
    int m_index;
    double m_gradient;
    static double learningRate; // learning rate closer to 0.0 is slower. learning rate of 0.2 is medium.
    static double alpha; // alpha is the momentum at which it learns. 0.5 is medium.
};

class TrainingData {
public:
    TrainingData(const std::string dataName, const std::string targetName);
    bool isDataEOF(void) const {return m_trainingData.eof();}
    bool isTargetEOF(void) const {return m_trainingTargets.eof();}
    void populateInputs(std::vector<double> &inputs);
    void populateTargets(std::vector<double> &targets);
    void restart(void);
private:
    std::ifstream m_trainingData;
    std::ifstream m_trainingTargets;
};

class Net {
public:
    Net(const std::vector<int> &topology); // constructor which takes the shape of the network
    void feedForward(const std::vector<double> &inputs); // feeds the inputs through the network, updating neuron outputs
    void backProp(const std::vector<double> &targets); // calculate root mean square error of each output neuron and update weights
    void getResults(std::vector<double> &results) const; // populates results vector
    void train(const std::string dataName, const std::string targetName, const int iterations, const int outputRate, const bool verbose);
    void predict(const std::vector<double> &inputs, std::vector<double> &results);
    double getRecentAverageError(void) const { return m_recentAverageError; }
    std::string printShape(void); // returns the shape of the network
    std::string examineNeurons(void);
    void saveWeights(const std::string fileName); // right now the topology has to be the same as the original network
    void loadWeights(const std::string fileName); // right now the topology has to be the same as the original network
private:
    std::vector<Layer> m_layers; // m_layers[layer][neuron]
    double m_error;
    double m_recentAverageError;
    static double m_recentAverageSmoothingFactor;
    std::fstream m_saveFile;
};
