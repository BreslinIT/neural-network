#include <vector>
#include <iostream>
#include <random>
#include <cstdlib>
#include <string>
#include <assert.h>
#include <cmath>
#include <fstream>

class Neuron;

typedef std::vector<Neuron> Layer;

struct Connection {
    double weight;
    double deltaWeight;
}; // stores the weight and change of weight of the connection

class Neuron {
public:
    Neuron(const unsigned weights, const int index);
    std::string examineWeights(void); // prints the weight values and output value
    void setOutputVal(double outputVal) {m_outputVal = outputVal;}
    double getOutputVal(void) const {return m_outputVal;}
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

double Neuron::learningRate = 0.1;
double Neuron::alpha = 0.45;

Neuron::Neuron(const unsigned weights, const int index) {
    for (int i = 0; i < weights; i++) {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }

    m_index = index;
}

double Neuron::activationFunction(double sum) {
    return tanh(sum);
}

double Neuron::activationFunctionDerivative(double sum) {
    return 1-(sum*sum);
}

void Neuron::feedForward(Layer &prevLayer) {
    double sum = 0.0;

    for (Neuron i:prevLayer) {
        sum += i.getOutputVal()*i.m_outputWeights[m_index].weight;
    }
    
    m_outputVal = activationFunction(sum);
}

double Neuron::sumDOW(const Layer &nextLayer) {
    double sum = 0.0;
    for (int i = 0; i < nextLayer.size() - 1; i++) {
        sum += m_outputWeights[i].weight * nextLayer[i].m_gradient;
    }
    return sum;
}

void Neuron::updateInputWeights(Layer &prevLayer) {
    for (int i = 0; i < prevLayer.size(); i++) {
        Neuron &neuron = prevLayer[i];
        double oldDeltaWeight = neuron.m_outputWeights[m_index].deltaWeight;

        double newDeltaWeight = (learningRate * neuron.getOutputVal() * m_gradient) + (alpha * oldDeltaWeight);

        neuron.m_outputWeights[m_index].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_index].weight += newDeltaWeight;
    }
}

void Neuron::calcOutputGradients(double target) {
    double difference = target - m_outputVal;
    m_gradient = difference * activationFunctionDerivative(m_outputVal);
}

void Neuron::calcHiddenGradients(Layer &nextLayer) {
    double dow = sumDOW(nextLayer);
    m_gradient = dow * activationFunctionDerivative(m_outputVal);
}

std::string Neuron::examineWeights() {
    std::string output = "";
    output += "     Output value: " + std::to_string(m_outputVal) + "\n";
    output += "     Connections (weight deltaWeight):\n";
    for (Connection i:m_outputWeights) {
        output += "         " + std::to_string(i.weight) + " " + std::to_string(i.deltaWeight) + "\n";
    }
    return output;
}

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

TrainingData::TrainingData(const std::string dataName, const std::string targetName) {
    m_trainingData.open(dataName);
    m_trainingTargets.open(targetName);
}

void TrainingData::populateInputs(std::vector<double> &inputs) {
    inputs.clear();
    std::string line;
    std::string::size_type sz;
    getline(m_trainingData,line);
    while ((int)line[0] != 97 && !isDataEOF()) {
        inputs.push_back(std::stod(line,&sz));
        getline(m_trainingData,line);
    }
}

void TrainingData::populateTargets(std::vector<double> &targets) {
    targets.clear();
    std::string line;
    std::string::size_type sz;
    getline(m_trainingTargets,line);
    while ((int)line[0] != 97 && !isTargetEOF()) {
        targets.push_back(std::stod(line,&sz));
        getline(m_trainingTargets,line);
    }
}

void TrainingData::restart() {
    m_trainingData.clear();
    m_trainingData.seekg(0, std::ios::beg);

    m_trainingTargets.clear();
    m_trainingTargets.seekg(0, std::ios::beg);
}

class Net {
public:
    Net(const std::vector<unsigned> &topology); // constructor which takes the shape of the network
    void feedForward(const std::vector<double> &inputs); // feeds the inputs through the network, updating neuron outputs
    void backProp(const std::vector<double> &targets); // calculate root mean square error of each output neuron and update weights
    void getResults(std::vector<double> &results) const; // populates results vector
    void train(const std::string dataName, const std::string targetName, const int iterations, const int outputRate);
    double getRecentAverageError(void) const { return m_recentAverageError; }
    void printShape(void); // returns the shape of the network
    void examineNeurons(void);
private:
    std::vector<Layer> m_layers; // m_layers[layer][neuron]
    double m_error;
    double m_recentAverageError;
    static double m_recentAverageSmoothingFactor;
};

double Net::m_recentAverageSmoothingFactor = 10.0;

Net::Net(const std::vector<unsigned> &topology) {
    unsigned numLayers = topology.size();

    for (int i=0; i<numLayers; i++) {
        m_layers.push_back(Layer());
        for (int j = 0; j <= topology[i]; j++) {
            if (i<numLayers-1) {
                m_layers.back().push_back(Neuron(topology[i+1],j));
            } // if we are not on the final layer, the neuron will have n weights where n is the number of neurons in the next layer
            else {
                m_layers.back().push_back(Neuron(0,j));
            } // if we are on the final layer, the neuron will have 0 weights.
        }
        m_layers.back().back().setOutputVal(1.0);
    }
}

void Net::feedForward(const std::vector<double> &inputs) {
    assert(inputs.size() == m_layers[0].size()-1); // -1 to account for bias neuron

    // assign initial values to the first layer
    for (int i = 0; i < inputs.size(); i++) {
        m_layers[0][i].setOutputVal(inputs[i]);
    }

    // progagate forward
    for (int i = 1; i < m_layers.size(); i++) {
        Layer &prevLayer = m_layers[i-1];
        for (int j = 0; j < m_layers[i].size()-1; j++) {
            m_layers[i][j].feedForward(prevLayer);
        }
    }
}

void Net::backProp(const std::vector<double> &targets) {
    assert(targets.size() == m_layers.back().size()-1);
    // calculate root mean squared error
    m_error = 0.0;
    Layer &outputLayer = m_layers.back();
    for (int i = 0; i < targets.size(); i++) {
        double difference = targets[i] - outputLayer[i].getOutputVal();
        m_error += difference * difference;
    }
    m_error /= targets.size();
    m_error = sqrt(m_error);
    // update recent average error
    m_recentAverageError = (m_recentAverageError * m_recentAverageSmoothingFactor + m_error) / (m_recentAverageSmoothingFactor + 1.0);

    // calculate output gradients
    for (int i = 0; i < targets.size(); i++) {
        outputLayer[i].calcOutputGradients(targets[i]);
    }

    // calculate hidden layer gradients
    for (int i = m_layers.size() - 2; i > 0; i--) {
        Layer &hiddenLayer = m_layers[i];
        Layer &nextLayer = m_layers[i+1];
        for (int j = 0; j < hiddenLayer.size(); j++) {
            hiddenLayer[j].calcHiddenGradients(nextLayer);
        }
    }

    // update connection weights
    for (int i = m_layers.size() - 1; i > 0; i--) {
        Layer &layer = m_layers[i];
        Layer &prevLayer = m_layers[i-1];
        for (int j = 0; j < m_layers[i].size()-1; j++) {
            layer[j].updateInputWeights(prevLayer);
        }
    }
}

void Net::getResults(std::vector<double> &results) const {
    results.clear();

    for (int i = 0; i < m_layers.back().size()-1; i++) {
        results.push_back(m_layers.back()[i].getOutputVal());
    }
}

void Net::train(const std::string dataName, const std::string targetName, const int iterations, const int outputRate=10) {
    TrainingData data(dataName, targetName);

    std::vector<double> inputs;
    std::vector<double> targets;
    std::vector<double> results;

    for (int i = 0; i < iterations; i++) {
        if (data.isDataEOF() || data.isTargetEOF()) {
            data.restart();
        }
        data.populateInputs(inputs);
        feedForward(inputs);

        data.populateTargets(targets);
        backProp(targets);
        if (i%outputRate==0) {
            std::cout << "inputs: ";
            for (double i:inputs) {
                std::cout << i << " ";
            }
            std::cout << "\n";

            std::cout << "target: ";
            for (double i:targets) {
                std::cout << i << " ";
            }
            std::cout << "\n";

            std::cout << "output: ";
            getResults(results);
            for (double i:results) {
                std::cout << i << " ";
            }
            std::cout << "\n";

            std::cout << "recent average error: " << getRecentAverageError() << "\n";
            std::cout << "error: " << m_error << "\n";
            std::cout << "\n";
        }
    }
}

void Net::printShape() {
    std::cout << "Number of layers: " << m_layers.size() << "\n";
    std::cout << "Layer sizes:";
    for (Layer i:m_layers) {
        std::cout << " " << i.size()-1;
    }
    std::cout << "\nThere is actually one more neuron is each layer; the bias neuron. This neuron does nothing in the last layer.\n";
}

void Net::examineNeurons() {
    for (int i = 0; i<m_layers.size(); i++) {
        if (i != m_layers.size()-1) {
            std::cout << "Layer " << i+1 << ":\n";
        }
        else {
            std::cout << "Output layer:\n";
        }
        for (int j = 0; j < m_layers[i].size(); j++) {
            if (j != m_layers[i].size()-1) {
                std::cout << "  Neuron " << j+1 << ":\n" << m_layers[i][j].examineWeights();
            }
            else {
                std::cout << "  Bias Neuron:\n" << m_layers[i][j].examineWeights();
            }
        }
    }
}

int main() {
    std::vector<unsigned> topology;
    topology.push_back(1);
    topology.push_back(4);
    topology.push_back(1);
    Net new_net(topology);

    new_net.train("testData.txt", "testTargets.txt",2000,1);
}