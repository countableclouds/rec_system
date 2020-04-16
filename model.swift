import TensorFlow

func normalize(tensor: Tensor<Float>) -> Tensor<Float>{
    print("\(tensor),\n\n \(tensor/Tensor.sqrt(tensor.squared().sum()))\n\n")
    return tensor/Tensor.sqrt(tensor.squared().sum())
}

public struct TrainingModel: Layer {
    var userLayer1: Dense<Float>
    var userLayer2: Dense<Float>
    var userLayer3: Dense<Float>
    var problemLayer1: Dense<Float>
    var problemLayer2: Dense<Float>
    var problemLayer3: Dense<Float>
    

    public init(userLength: Int, problemLength: Int, vectorizationLength: Int, hiddenSize: Int) {
        userLayer1 = Dense<Float>(inputSize: userLength, outputSize: hiddenSize, activation: relu)
        userLayer2 = Dense<Float>(inputSize: hiddenSize, outputSize: hiddenSize, activation: relu)
        userLayer3 = Dense<Float>(inputSize: hiddenSize, outputSize: vectorizationLength, activation: sigmoid)
        problemLayer1 = Dense<Float>(inputSize: problemLength, outputSize: hiddenSize, activation: relu)
        problemLayer2 = Dense<Float>(inputSize: hiddenSize, outputSize: hiddenSize, activation: relu)
        problemLayer3 = Dense<Float>(inputSize: hiddenSize, outputSize: vectorizationLength, activation: sigmoid)
    }

    @differentiable
    public func callAsFunction(_ input: [Tensor<Float>]) -> Tensor<Float> {
        let firstTensor = input[0].sequenced(through: userLayer1, userLayer2, userLayer3).batchNormalized(alongAxis: 0)
        let secondTensor = input[1].sequenced(through: problemLayer1, problemLayer2, problemLayer3).batchNormalized(alongAxis: 1)
        let tensor = firstTensor • secondTensor
        return max(tensor, 1e-5)
    }
}

public class Model {
    public let ratingMatrix: Tensor<Bool>
    public let zeroIndices: Tensor<Int64>
    public let ratingIndices: Tensor<Int64>
    public var classifier: TrainingModel
    public var optimizer: Adam<TrainingModel>

    public init(ratingMat: Tensor<Bool>) {
        ratingMatrix = ratingMat
        ratingIndices = ratingMat.nonZeroIndices()
        zeroIndices = ratingMatrix.elementsLogicalNot().nonZeroIndices()
        classifier = TrainingModel(userLength: ratingMatrix.shape[1], problemLength: ratingMatrix.shape[0], vectorizationLength: 4, hiddenSize: 4)
        optimizer = Adam(for: classifier, learningRate: 0.02)
        Context.local.learningPhase = .training

        
    }

    public func getInstances(negRatio: Float) -> (userTrain: Tensor<Int64>, problemTrain: Tensor<Int64>, ratingTrain: Tensor<Float>){
        var maskArray: [Bool] = [];
        let dim = ratingMatrix.shape
        for _ in 0..<dim[0] {
            maskArray.append(Float.random(in: 0..<1) < negRatio)
        }
        let mask = Tensor<Bool>(maskArray)
        let trainZeroIndices = zeroIndices.gathering(where: mask)
        let trainIndices = trainZeroIndices ++ ratingIndices
        let splitted = trainIndices.split(count: 2, alongAxis: 1)
        var userTrain = splitted[0].transposed()[0]
        var problemTrain = splitted[1].transposed()[0]
        let numProblems = problemTrain.shape[0]
        var atIndices: [Int64] = []
        for i in 0..<numProblems {
            atIndices.append(Int64(userTrain[i])!*Int64(dim[0])  + Int64(problemTrain[i])!)
        }
        var ratingTrain = Tensor<Float>(ratingMatrix).flattened().gathering(atIndices: Tensor<Int64>(atIndices), alongAxis: 0)
        let shuffledIndices: [Int64] = (0..<Int64(numProblems)).shuffled()
        userTrain = userTrain.gathering(atIndices: Tensor<Int64>(shuffledIndices), alongAxis: 0)
        problemTrain = problemTrain.gathering(atIndices: Tensor<Int64>(shuffledIndices), alongAxis: 0)
        ratingTrain = ratingTrain.gathering(atIndices: Tensor<Int64>(shuffledIndices), alongAxis: 0)
        return (userTrain, problemTrain, ratingTrain)
    }

    public func runEpoch(negRatio: Float, batchSize: Int) {
        let (userTrain, problemTrain, ratingTrain) = getInstances(negRatio: negRatio)
        let trainLen = userTrain.shape[0]
        let numBatches = trainLen/batchSize
        for i in 0...numBatches {
            let minIdx = i * batchSize
            let maxIdx = min((i+1)*batchSize, trainLen)
            let userTrainBatch = userTrain[minIdx..<maxIdx]
            let userMatrixInput = Tensor<Float>(ratingMatrix.gathering(atIndices: userTrainBatch, alongAxis: 0))
            let problemTrainBatch = problemTrain[minIdx..<maxIdx]
            let problemMatrixInput = Tensor<Float>(ratingMatrix.gathering(atIndices: problemTrainBatch, alongAxis: 1)).transposed()
            let y = ratingTrain[minIdx..<maxIdx]
            let 𝛁model = gradient(at: classifier) { classifier -> Tensor<Float> in
                let ŷ: Tensor<Float> = classifier.callAsFunction([userMatrixInput, problemMatrixInput])
                let first_loss = (log(ŷ)*y).sum()
                let second_loss = (log(1-ŷ)*(1-y)).sum()
                let loss = -first_loss-second_loss
                print("Loss: \(loss)")
                return loss 
            }
            optimizer.update(&classifier, along: 𝛁model)
        

        }
    }


    public func train(numEpochs: UInt32, negRatio: Float, batchSize: Int){
        print("Training beginning")
        for _ in 1...numEpochs {
            runEpoch(negRatio: negRatio, batchSize: batchSize)
        }
    }


    // public func 
}
let arr = Tensor<Bool>(arrayLiteral: [false, false, true], [true, false, true], [true, true, false], [false, false, false])
var model = Model(ratingMat: arr)
model.train(numEpochs: 100, negRatio: 0.5, batchSize: 10)