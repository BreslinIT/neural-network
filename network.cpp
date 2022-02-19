#include <vector>
#include <iostream>
#include <random>
#include <cstdlib>
#include <string>
#include <assert.h>
#include <cmath>
#include <fstream>
#include <time.h>
#include <experimental/filesystem>
#include "network.h"


Neuron::Neuron(const int weights, const int index, double learningRate, double alpha) {
    m_learningRate = learningRate;
    m_alpha = alpha;
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
    return 2/(1 + cosh(2*sum));
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

        double newDeltaWeight = (m_learningRate * neuron.getOutputVal() * m_gradient) + (m_alpha * oldDeltaWeight);

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
    output += std::to_string(m_outputVal) + "\n";
    output += std::to_string(m_gradient) + "\n";
    for (Connection i:m_outputWeights) {
        output += std::to_string(i.weight) + "\n" + std::to_string(i.deltaWeight) + "\n";
    }
    return output;
}

TrainingData::TrainingData(const std::string dataName, const std::string targetName) {
    m_trainingData.open(dataName);
    m_trainingTargets.open(targetName);
}

void TrainingData::populateInputs(std::vector<double> &inputs) {
    inputs.clear();
    std::string line;
    std::string::size_type sz;
    getline(m_trainingData,line);
    while ((int)line[0] != 47 && !isDataEOF()) {
        inputs.push_back(std::stod(line,&sz));
        getline(m_trainingData,line);
    }
}

void TrainingData::populateTargets(std::vector<double> &targets) {
    targets.clear();
    std::string line;
    std::string::size_type sz;
    getline(m_trainingTargets,line);
    while ((int)line[0] != 47 && !isTargetEOF()) {
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

double Net::m_recentAverageSmoothingFactor = 10.0;

Net::Net(const std::vector<int> &topology, double learningRate=0.2, double alpha=0.5) {
    srand(time(NULL));
    int numLayers = topology.size();

    for (int i=0; i<numLayers; i++) {
        m_layers.push_back(Layer());
        for (int j = 0; j <= topology[i]; j++) {
            if (i<numLayers-1) {
                m_layers.back().push_back(Neuron(topology[i+1],j,learningRate,alpha));
            } // if we are not on the final layer, the neuron will have n weights where n is the number of neurons in the next layer
            else {
                m_layers.back().push_back(Neuron(0,j,learningRate,alpha));
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

void Net::train(const std::string dataName, const std::string targetName, const int iterations, const bool verbose=true, const bool createGif=false, const int outputRate=1) {
    int frames = 1;
    if (createGif) {
        if (iterations > 300) {
            frames = iterations/300;
        }
        for (const auto& entry : std::experimental::filesystem::directory_iterator("weights")) {
            std::experimental::filesystem::remove(entry.path());
        }
    }
    
    
    TrainingData data(dataName, targetName);

    std::vector<double> inputs;
    std::vector<double> targets;
    std::vector<double> results;

    for (int iteration = 0; iteration < iterations; iteration++) {
        if (data.isDataEOF() || data.isTargetEOF()) {
            data.restart();
        }
        data.populateInputs(inputs);
        feedForward(inputs);

        data.populateTargets(targets);
        backProp(targets);
        if (verbose){
            if (iteration%outputRate==0) {
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
                std::cout << "total iterations: " << iteration+1 << "\n";
                std::cout << "\n";
            }
        }
        if (createGif) {
            if (iteration%frames == 0) {
                std::string fileName = "weights/" + std::to_string(iteration) + ".weights";
                saveWeights(fileName);
            }
        }
    }
}

void Net::predict(const std::vector<double> &inputs, std::vector<double> &results) {
    feedForward(inputs);
    getResults(results);
}

std::string Net::printShape() {
    std::string output = "";
    output += std::to_string(m_layers.size()) + "\n";
    for (Layer i:m_layers) {
        output += std::to_string(i.size()) + "\n";
    }
    return output;
}

std::string Net::examineNeurons() {
    std::string output = "";
    for (int i = 0; i<m_layers.size(); i++) {
        for (int j = 0; j < m_layers[i].size(); j++) {
            output += m_layers[i][j].examineWeights();
        }
    }
    return output;
}

void Net::saveWeights(const std::string fileName) {
    m_saveFile.open(fileName,std::fstream::out);
    m_saveFile << printShape();
    m_saveFile << examineNeurons();
    m_saveFile.close();
}
void Net::loadWeights(const std::string fileName) {
    m_saveFile.open(fileName);
    std::string line;
    std::string::size_type sz;

    getline(m_saveFile,line);
    int layers = stoi(line);
    assert(layers == m_layers.size());
    int neurons;
    std::vector<int> topology;
    for (int i = 0; i < layers; i++) {
        getline(m_saveFile,line);
        neurons = stoi(line);
        assert(neurons == m_layers[i].size());
        topology.push_back(neurons);
    }

    double weight;
    for (int i = 0; i < topology.size(); i++) {
        for (int j = 0; j < topology[i]; j++) {
            getline(m_saveFile,line);
            weight = stod(line,&sz);
            m_layers[i][j].setOutputVal(weight); // this is the output val

            getline(m_saveFile,line);
            weight = stod(line,&sz);
            m_layers[i][j].setGradient(weight); // this is the gradient
            if (i != topology.size()) {
                m_layers[i][j].resetConnections();
                for (int k = 0; k < topology[i+1]-1; k++) {
                    Connection weights;
                    getline(m_saveFile,line);
                    weight = stod(line,&sz);
                    weights.weight = weight;

                    getline(m_saveFile,line);
                    weight = stod(line,&sz);
                    weights.deltaWeight = weight;

                    m_layers[i][j].addConnection(weights);
                }  // num neurons in next layer to determine connections
            }
        }
    }


    m_saveFile.close();
}

int main() {
    std::vector<int> topology;
    topology.push_back(2);
    topology.push_back(4);
    topology.push_back(3);
    topology.push_back(1);
    Net new_net(topology);

    new_net.train("testData.txt", "testTargets.txt",2000,true,true);
    
    // new_net.loadWeights("weights.weights");
    // std::vector<double> test;
    // test.push_back(0.0);
    // test.push_back(1.0);
    // new_net.feedForward(test);
    new_net.saveWeights("weights.weights");
}